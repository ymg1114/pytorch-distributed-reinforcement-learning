import os
import zmq

import torch
import torch.nn.functional as F

from torch.distributions import Categorical
from functools import partial

from utils.utils import Protocol, encode, make_gpu_batch
from torch.optim import Adam

from tensorboardX import SummaryWriter

L_IP = "127.0.0.1" # 동일 서브넷 다른 머신 사용 가능.


def compute_gae(
    delta,
    gamma,
    lambda_,
):
    """
    Compute GAE.

    delta: TD-errors, shape of (batch, seq, dim)
    gamma: Discount factor.
    lambda_: GAE lambda parameter.
    """

    gae = 0
    returns = []
    for step in reversed(range(delta.shape[1])):
        d = delta[:, step]
        gae = d + gamma * lambda_ * gae
        returns.insert(0, gae)

    return torch.stack(returns, dim=1)


def compute_vtrace_targets(rewards, values, bootstrap_value, gamma, lambda_, rho_cap=1.0, c_cap=1.0):
    """
    Compute V-trace targets for advantages and values.
    """
    
    num_steps = rewards.shape[1] - 1 # batch, seq, dim
    advantages = torch.zeros_like(rewards)
    corrected_values = torch.zeros_like(values)
    corrected_values[:, -1] = bootstrap_value
    
    for t in reversed(range(num_steps)):
        # Importance sampling ratio 계산 (실제 구현에서는 behavior와 target policy 비율을 사용해야 함)
        rho = min(rho_cap, 1.0)  # 예시에서는 모든 rho를 1로 가정
        
        # Truncated importance weight 계산
        c = min(c_cap, 1.0)  # 예시에서는 모든 c를 1로 가정

        # V-trace corrected TD-error
        delta = rho * (rewards[:, t] + gamma * corrected_values[:, t + 1] - values[:, t])
        
        # V-trace corrected advantage 계산
        advantages[:, t] = delta + gamma * lambda_ * c * (advantages[:, t + 1] if t + 1 < num_steps else 0)

        # V-trace corrected value 계산
        corrected_values[:, t] = values[:, t] + advantages[:, t]

    return advantages.detach(), corrected_values.detach()


class Learner:
    def __init__(self, args, mutex, model, queue):
        self.args = args
        self.mutex = mutex
        self.batch_queue = queue

        self.device = self.args.device
        self.model = model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.ct = Categorical

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set()
        self.mean_cal_interval = 30
        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log

    def __del__(self): # 소멸자
        self.pub_socket.close()

    def zeromq_set(self):
        context = zmq.Context()

        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{L_IP}:{self.args.learner_port+1}"
        )  # publish fresh learner-model

    def pub_model(self, model_state_dict):
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_state_dict)])

    # TODO: 현재 작동하지 않아 주석 상태. 개선 필요.
    def log_loss_tensorboard(self, loss, detached_losses):
        self.writer.add_scalar("total-loss", float(loss.item()), self.idx)
        self.writer.add_scalar(
            "original-value-loss", detached_losses["value-loss"], self.idx
        )
        self.writer.add_scalar(
            "original-policy-loss", detached_losses["policy-loss"], self.idx
        )
        self.writer.add_scalar(
            "original-policy-entropy", detached_losses["policy-entropy"], self.idx
        )

    # PPO
    def learning(self):
        self.idx = 0

        while True:
            batch_args = None
            with self.mutex.lock():
                batch_args = self.batch_queue.get()

            if batch_args is not None:
                # Basically, mini-batch-learning (batch, seq, feat)
                obs, act, rew, logits, is_fir, hx, cx = self.to_gpu(*batch_args)
                behav_log_probs = (
                    self.ct(F.softmax(logits, dim=-1))
                    .log_prob(act.squeeze(-1))
                    .unsqueeze(-1)
                )

                # epoch-learning
                for _ in range(self.args.K_epoch):
                    lstm_states = (
                        hx[:, 0],
                        cx[:, 0],
                    )  # (batch, seq, hidden) -> (batch, hidden)

                    # on-line model forwarding
                    log_probs, entropy, value = self.model(
                        obs,  # (batch, seq, *sha)
                        lstm_states,  # ((batch, hidden), (batch, hidden))
                        act,  # (batch, seq, 1)
                    )

                    td_target = (
                        rew[:, :-1]
                        + self.args.gamma * (1 - is_fir[:, 1:]) * value[:, 1:]
                    )
                    delta = td_target - value[:, :-1]
                    delta = delta.cpu().detach()

                    gae = compute_gae(
                        delta, self.args.gamma, self.args.lmbda
                    )  # ppo-gae (advantage)

                    ratio = torch.exp(
                        log_probs[:, :-1] - behav_log_probs[:, :-1]
                    )  # a/b == log(exp(a)-exp(b))
                    surr1 = ratio * gae
                    surr2 = (
                        torch.clamp(
                            ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                        )
                        * gae
                    )

                    loss_policy = -torch.min(surr1, surr2).mean()
                    loss_value = F.smooth_l1_loss(
                        value[:, :-1], td_target.detach()
                    ).mean()
                    policy_entropy = entropy[:, :-1].mean()

                    loss = (
                        self.args.policy_loss_coef * loss_policy
                        + self.args.value_loss_coef * loss_value
                        - self.args.entropy_coef * policy_entropy
                    )

                    detached_losses = {
                        "policy-loss": loss_policy.detach().cpu(),
                        "value-loss": loss_value.detach().cpu(),
                        "policy-entropy": policy_entropy.detach().cpu(),
                    }

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    print(
                        "loss {:.5f} original_value_loss {:.5f} original_policy_loss {:.5f} original_policy_entropy {:.5f}".format(
                            loss.item(),
                            detached_losses["value-loss"],
                            detached_losses["policy-loss"],
                            detached_losses["policy-entropy"],
                        )
                    )
                    self.optimizer.step()

                self.pub_model(self.model.state_dict())

                # if (self.idx % self.args.loss_log_interval == 0):
                #     self.log_loss_tensorboard(loss, detached_losses)

                if self.idx % self.args.model_save_interval == 0:
                    torch.save(
                        self.model,
                        os.path.join(self.args.model_dir, f"ppo_{self.idx}.pt"),
                    )

                self.idx += 1
                
    # # IMPALA
    # def learning(self):
    #     self.idx = 0

    #     while True:
    #         batch_args = None
    #         with self.mutex.lock():
    #             batch_args = self.batch_queue.get()

    #         if batch_args is not None:
    #             # Basically, mini-batch-learning (batch, seq, feat)
    #             obs, act, rew, logits, is_fir, hx, cx = self.to_gpu(*batch_args)
    #             behav_log_probs = (
    #                 self.ct(F.softmax(logits, dim=-1))
    #                 .log_prob(act.squeeze(-1))
    #                 .unsqueeze(-1)
    #             )

    #             # epoch-learning
    #             for _ in range(self.args.K_epoch):
    #                 lstm_states = (
    #                     hx[:, 0],
    #                     cx[:, 0],
    #                 )  # (batch, seq, hidden) -> (batch, hidden)

    #                 # on-line model forwarding
    #                 log_probs, entropy, value = self.model(
    #                     obs,  # (batch, seq, *sha)
    #                     lstm_states,  # ((batch, hidden), (batch, hidden))
    #                     act,  # (batch, seq, 1)
    #                 )

    #                 # V-trace를 사용하여 off-policy corrections 연산
    #                 advantages, corrected_values = compute_vtrace_targets(
    #                     rewards=rew,
    #                     values=value,
    #                     bootstrap_value=value[:, -1],
    #                     gamma=self.args.gamma,
    #                     lambda_=self.args.lmbda,
    #                     )
                    
    #                 loss_policy = -torch.mean(log_probs[:, :-1] * advantages[:, :-1]).mean()
    #                 loss_value = F.smooth_l1_loss(value[:, :-1], corrected_values[:, :-1]).mean()
    #                 policy_entropy = entropy[:, :-1].mean()

    #                 loss = (
    #                     self.args.policy_loss_coef * loss_policy
    #                     + self.args.value_loss_coef * loss_value
    #                     - self.args.entropy_coef * policy_entropy
    #                 )

    #                 detached_losses = {
    #                     "policy-loss": loss_policy.detach().cpu(),
    #                     "value-loss": loss_value.detach().cpu(),
    #                     "policy-entropy": policy_entropy.detach().cpu(),
    #                 }

    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(), self.args.max_grad_norm
    #                 )
    #                 print(
    #                     "loss {:.5f} original_value_loss {:.5f} original_policy_loss {:.5f} original_policy_entropy {:.5f}".format(
    #                         loss.item(),
    #                         detached_losses["value-loss"],
    #                         detached_losses["policy-loss"],
    #                         detached_losses["policy-entropy"],
    #                     )
    #                 )
    #                 self.optimizer.step()

    #             self.pub_model(self.model.state_dict())

    #             # if (self.idx % self.args.loss_log_interval == 0):
    #             #     self.log_loss_tensorboard(loss, detached_losses)

    #             if self.idx % self.args.model_save_interval == 0:
    #                 torch.save(
    #                     self.model,
    #                     os.path.join(self.args.model_dir, f"ppo_{self.idx}.pt"),
    #                 )

    #             self.idx += 1                