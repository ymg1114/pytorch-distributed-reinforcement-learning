import os
import zmq
import time
import queue
import torch
import torch.nn.functional as F
import multiprocessing as mp

from torch.distributions import Categorical
from functools import partial

from utils.lock import Mutex
from utils.utils import Protocol, encode, make_gpu_batch, writer, L_IP, ExecutionTimer, Params
from torch.optim import Adam, RMSprop


timer = ExecutionTimer(num_transition=Params.seq_len*Params.batch_size*1) # Learner에서 데이터 처리량 (학습)


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


def compute_v_trace(behav_log_probs, target_log_probs, is_fir, rewards, values, gamma, rho_bar=0.8, c_bar=1.0):
    # Importance sampling weights (rho)
    rho = torch.exp(target_log_probs[:, :-1] - behav_log_probs[:, :-1]).detach() # a/b == exp(log(a)-log(b))
    # rho_clipped = torch.clamp(rho, max=rho_bar)
    rho_clipped = torch.clamp(rho, min=0.1, max=rho_bar)
    
    # truncated importance weights (c)
    c = torch.exp(target_log_probs[:, :-1] - behav_log_probs[:, :-1]).detach() # a/b == exp(log(a)-log(b))
    c_clipped = torch.clamp(c, max=c_bar)

    td_target = (
        rewards[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * values[:, 1:]
    )    
    deltas = rho_clipped * (td_target - values[:, :-1]) # TD-Error with 보정
    vs_minus_v_xs = torch.zeros_like(values)
    for t in reversed(range(deltas.size(1))):
        vs_minus_v_xs[:, t] = deltas[:, t] + (gamma * (1 - is_fir))[:, t + 1] * c_clipped[:, t] * vs_minus_v_xs[:, t + 1]
    
    values_target = values + vs_minus_v_xs # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    advantages = rho_clipped * (rewards[:, :-1] + gamma * (1 - is_fir[:, 1:]) * values_target[:, 1:] - values[:, :-1])
    
    return rho_clipped, advantages.detach(), values_target.detach()


class Learner:
    def __init__(self, args, mutex, model, queue, stat_queue=None, heartbeat=None):
        self.args = args
        self.mutex: Mutex = mutex
        self.batch_queue = queue
        self.stat_queue = stat_queue
        self.heartbeat = heartbeat
        
        self.device = self.args.device
        self.model = model.to(self.device)

        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        self.CT = Categorical

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.stat_list = []
        self.zeromq_set()
        self.writer = writer
        
    def __del__(self): # 소멸자
        self.pub_socket.close()

    def zeromq_set(self):
        context = zmq.Context()

        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{L_IP}:{self.args.learner_port+1}"
        )  # publish fresh learner-model

    def pub_model(self, model_state_dict): # learner -> worker
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_state_dict)])
            
    def log_loss_tensorboard(self, timer, loss, detached_losses):
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
        self.writer.add_scalar(
            "min-ratio", detached_losses["ratio"].min(), self.idx
        )
        self.writer.add_scalar(
            "max-ratio", detached_losses["ratio"].max(), self.idx
        )
        self.writer.add_scalar(
            "avg-ratio", detached_losses["ratio"].mean(), self.idx
        )      
        
        if self.stat_queue is not None and isinstance(self.stat_queue, mp.queues.Queue):
            try:
                stat_dict = self.stat_queue.get_nowait()
                assert "tag" in stat_dict
                assert "x" in stat_dict
                assert "y" in stat_dict
                
                #TODO: 좋은 형태의 구조는 아님. LearnerStorage 쓰루풋을 텐서보드에 기록하기 위함.
                x = self.idx if "learner-storage-throughput" in stat_dict["tag"] else stat_dict["x"]
                self.writer.add_scalar(stat_dict["tag"], stat_dict["y"], x)
                
            except queue.Empty:
                # 큐가 비어있음을 처리
                #TODO: 좋은 형태의 구조는 아님
                print("stat_queue is empty.")

        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.add_scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.add_scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )
                        
    # PPO
    def learning_ppo(self):
        self.idx = 0

        while True:
            batch_args = None
            with timer.timer("learner-throughput", check_throughput=True):
                with timer.timer("learner-batching-time"):
                    batch_args = self.batch_queue.get()
                    # batch_args = self.mutex.get(self.batch_queue)
                    # batch_args = self.mutex.get_with_timeout(self.batch_queue)
                    
                if batch_args is not None:
                    with timer.timer("learner-forward-time"):
                        # Basically, mini-batch-learning (batch, seq, feat)
                        obs, act, rew, logits, is_fir, hx, cx = self.to_gpu(*batch_args)
                        behav_log_probs = (
                            self.CT(F.softmax(logits, dim=-1))
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
                            )  # a/b == exp(log(a)-log(b))
                                                
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
                                "ratio": ratio.detach().cpu(),
                            }
                            
                    with timer.timer("learner-backward-time"):
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        print(
                            "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                                loss.item(),
                                detached_losses["value-loss"],
                                detached_losses["policy-loss"],
                                detached_losses["policy-entropy"],
                                detached_losses["ratio"].mean(),
                            )
                        )
                        self.optimizer.step()

                    self.pub_model(self.model.state_dict())

                    if (self.idx % self.args.loss_log_interval == 0):
                        self.log_loss_tensorboard(timer, loss, detached_losses)
                        
                    if self.idx % self.args.model_save_interval == 0:
                        torch.save(
                            self.model,
                            os.path.join(self.args.model_dir, f"{self.args.algo}_{self.idx}.pt"),
                        )

                    self.idx += 1
                    
                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()
    # IMPALA
    def learning_impala(self):
        self.idx = 0

        while True:
            batch_args = None
            with timer.timer("learner-throughput", check_throughput=True):
                with timer.timer("learner-batching-time"):
                    batch_args = self.batch_queue.get()
                    # batch_args = self.mutex.get(self.batch_queue)
                    # batch_args = self.mutex.get_with_timeout(self.batch_queue)
                    
                if batch_args is not None:
                    with timer.timer("learner-forward-time"):
                        # Basically, mini-batch-learning (batch, seq, feat)
                        obs, act, rew, logits, is_fir, hx, cx = self.to_gpu(*batch_args)
                        behav_log_probs = (
                            self.CT(F.softmax(logits, dim=-1))
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
                            
                            # V-trace를 사용하여 off-policy corrections 연산
                            ratio, advantages, values_target = compute_v_trace(
                                behav_log_probs=behav_log_probs, 
                                target_log_probs=log_probs,
                                is_fir=is_fir,
                                rewards=rew,
                                values=value, 
                                gamma=self.args.gamma,
                                )
                            
                            loss_policy = -(log_probs[:, :-1] * advantages).mean()
                            loss_value = F.smooth_l1_loss(value[:, :-1], values_target[:, :-1]).mean()
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
                                "ratio": ratio.detach().cpu(),
                            }

                    with timer.timer("learner-backward-time"):
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        print(
                            "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                                loss.item(),
                                detached_losses["value-loss"],
                                detached_losses["policy-loss"],
                                detached_losses["policy-entropy"],
                                detached_losses["ratio"].mean(),
                            )
                        )
                        self.optimizer.step()

                    self.pub_model(self.model.state_dict())

                    if (self.idx % self.args.loss_log_interval == 0):
                        self.log_loss_tensorboard(timer, loss, detached_losses)
                        
                    if self.idx % self.args.model_save_interval == 0:
                        torch.save(
                            self.model,
                            os.path.join(self.args.model_dir, f"{self.args.algo}_{self.idx}.pt"),
                        )

                    self.idx += 1

                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()                   