import os, sys
import zmq
import asyncio
import torch
import pickle
import time
import numpy as np
from multiprocessing import shared_memory, Process, Queue
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.distributions import Categorical
from functools import partial
from collections import deque
from agents.learner_storage import LearnerStorage
from utils.utils import Protocol, encode, decode, make_gpu_batch
from utils.lock import Lock
from threading import Thread
from torch.optim import RMSprop, Adam

# from buffers.batch_buffer import LearnerBatchStorage
from tensorboardX import SummaryWriter

local = "127.0.0.1"

# L = Lock() # 사용하지 않는 코드


class Learner:
    def __init__(self, args, mutex, model, queue):
        self.args = args
        self.mutex = mutex
        self.batch_queue = queue

        self.device = self.args.device
        self.model = model.to(args.device)
        # self.model.share_memory() # make other processes can assess
        # self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.ct = Categorical

        # self.q_workers = mp.Queue(maxsize=args.batch_size) # q for multi-worker-rollout
        # self.batch_buffer = LearnerBatchStorage(args, obs_shape)

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set()
        self.mean_cal_interval = 30
        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log

    def zeromq_set(self):
        context = zmq.Context()

        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{local}:{self.args.learner_port+1}"
        )  # publish fresh learner-model

    # 사용하지 않는 코드
    # def data_subscriber(self, q_batchs):
    #     self.l_t = Thread(target=self.receive_data, args=(q_batchs,), daemon=True)
    #     self.l_t.start()

    # 사용하지 않는 코드
    # def receive_data(self, q_batchs):
    #     while True:
    #         protocol, data = decode(*self.sub_socket.recv_multipart())
    #         if protocol is Protocol.Batch:
    #             if q_batchs.qsize() == q_batchs._maxsize:
    #                 L.get(q_batchs)
    #             L.put(q_batchs, data)

    #         elif protocol is Protocol.Stat:
    #             self.stat_list.append(data["mean_stat"])
    #             if len(self.stat_list) >= self.stat_log_len:
    #                 mean_stat = self.process_stat()
    #                 self.log_stat_tensorboard({"log_len": self.stat_log_len, "mean_stat": mean_stat})
    #                 self.stat_list = []
    #         else:
    #             assert False, f"Wrong protocol: {protocol}"

    #         time.sleep(0.01)

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
                # Basically, mini-batch-learning (seq, batch, feat)
                obs, act, rew, logits, is_fir, hx, cx = self.to_gpu(*batch_args)
                behav_log_probs = (
                    self.ct(F.softmax(logits, dim=-1))
                    .log_prob(act.squeeze(-1))
                    .unsqueeze(-1)
                )

                # epoch-learning
                for _ in range(self.args.K_epoch):
                    lstm_states = (
                        hx[0],
                        cx[0],
                    )  # (seq, batch, hidden) -> (batch, hidden)

                    # on-line model forwarding
                    log_probs, entropy, value = self.model(
                        obs,  # (seq, batch, *sha)
                        lstm_states,  # ((batch, hidden), (batch, hidden))
                        act,  # (seq, batch, 1)
                    )

                    td_target = (
                        rew[:-1] + self.args.gamma * (1 - is_fir[1:]) * value[1:]
                    )
                    delta = td_target - value[:-1]
                    delta = delta.cpu().detach().numpy()

                    # ppo-gae (advantage)
                    advantages = []
                    advantage_t = np.zeros(
                        delta.shape[1:]
                    )  # Terminal: (seq, batch, d) -> (batch, d)
                    for delta_row in delta[::-1]:
                        advantage_t = (
                            delta_row + self.args.gamma * self.args.lmbda * advantage_t
                        )  # recursive
                        advantages.append(advantage_t)
                    advantages.reverse()
                    # advantage = torch.stack(advantages, dim=0).to(torch.float)
                    advantage = torch.tensor(advantages, dtype=torch.float).to(
                        self.device
                    )

                    ratio = torch.exp(
                        log_probs[:-1] - behav_log_probs[:-1]
                    )  # a/b == log(exp(a)-exp(b))
                    surr1 = ratio * advantage
                    surr2 = (
                        torch.clamp(
                            ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip
                        )
                        * advantage
                    )

                    loss_policy = -torch.min(surr1, surr2).mean()
                    loss_value = F.smooth_l1_loss(value[:-1], td_target.detach()).mean()
                    policy_entropy = entropy[:-1].mean()

                    # Summing all the losses together
                    loss = (
                        self.args.policy_loss_coef * loss_policy
                        + self.args.value_loss_coef * loss_value
                        - self.args.entropy_coef * policy_entropy
                    )

                    # These are only used for the statistics
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
                        os.path.join(self.args.model_dir, f"impala_{self.idx}.pt"),
                    )

                self.idx += 1
