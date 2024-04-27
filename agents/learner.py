import os
import zmq
import time
import torch
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np

from torch.distributions import Categorical
from functools import partial

from utils.lock import Mutex
from utils.utils import Protocol, encode, make_gpu_batch, L_IP, ExecutionTimer, Params
from torch.optim import Adam, RMSprop

from . import ppo_wrapper, impala_wrapper, sac_wrapper

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class Learner:
    def __init__(
        self, args, mutex, model, queue, shared_stat_array=None, heartbeat=None
    ):
        self.args = args
        self.mutex: Mutex = mutex
        self.batch_queue = queue

        if shared_stat_array is not None:
            self.np_shared_stat_array: np.ndarray = np.frombuffer(
                buffer=shared_stat_array.get_obj(), dtype=np.float64, count=-1
            )

        self.heartbeat = heartbeat

        self.device = self.args.device
        self.model = model.to(self.device)

        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        self.CT = Categorical

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.zeromq_set()
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log

    def __del__(self):  # 소멸자
        self.pub_socket.close()

    def zeromq_set(self):
        context = zmq.Context()

        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{L_IP}:{self.args.learner_port+1}"
        )  # publish fresh learner-model

    def pub_model(self, model_state_dict):  # learner -> worker
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_state_dict)])

    def log_loss_tensorboard(self, timer: ExecutionTimer, loss, detached_losses):
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
        self.writer.add_scalar("min-ratio", detached_losses["ratio"].min(), self.idx)
        self.writer.add_scalar("max-ratio", detached_losses["ratio"].max(), self.idx)
        self.writer.add_scalar("avg-ratio", detached_losses["ratio"].mean(), self.idx)

        # TODO: 좋은 형태의 구조는 아님
        if self.np_shared_stat_array is not None:
            assert self.np_shared_stat_array.size == 3
            if (
                bool(self.np_shared_stat_array[2]) is True
            ):  # 기록 가능 활성화 (activate)

                x = self.np_shared_stat_array[0]  # global game counts
                y = self.np_shared_stat_array[1]  # mean-epi-rew

                self.writer.add_scalar("50-game-mean-stat-of-epi-rew", y, x)

                self.np_shared_stat_array[2] = 0  # 기록 가능 비활성화 (deactivate)

        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.add_scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.add_scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )

    @ppo_wrapper(timer=timer)
    def learning_ppo(self): ...

    @impala_wrapper(timer=timer)
    def learning_impala(self): ...

    @sac_wrapper(timer=timer)
    def learning_sac(self): ...
