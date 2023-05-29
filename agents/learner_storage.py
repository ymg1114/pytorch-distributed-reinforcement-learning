import torch
import zmq
import zmq.asyncio
import asyncio
import time
import numpy as np
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

from buffers.rollout_assembler import RolloutAssembler
from utils.utils import Protocol, mul, encode, decode, flatten, to_torch, counted
from utils.lock import Lock
from threading import Thread
from tensorboardX import SummaryWriter

local = "127.0.0.1"


class LearnerStorage:
    def __init__(self, args, mutex, dataframe_keyword, queue, obs_shape):
        self.args = args
        self.dataframe_keyword = dataframe_keyword
        self.obs_shape = obs_shape
        # self.device = self.args.device
        self.device = torch.device("cpu")

        self.stat_list = []
        self.stat_log_len = 20
        self.mutex = mutex
        self._init = False

        self.batch_queue = queue

        self.zeromq_set()
        self.init_shared_memory()
        self.rollout_assembler = RolloutAssembler(args)
        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log

    # def __del__(self):
    #     if hasattr(self, "_init") and self._init is True:
    #         def del_sh(shm):
    #             shm.close()
    #             shm.unlink()
    #         del_sh(self.sh_obs_batch)
    #         del_sh(self.sh_act_batch)
    #         del_sh(self.sh_rew_batch)
    #         del_sh(self.sh_logits_batch)
    #         del_sh(self.sh_is_fir_batch)
    #         del_sh(self.sh_hx_batch)
    #         del_sh(self.sh_cx_batch)
    #         del_sh(self.sh_data_num)

    def zeromq_set(self):
        context = zmq.asyncio.Context()

        # learner-storage <-> manager
        self.sub_socket = context.socket(zmq.SUB)  # subscribe batch-data, stat-data
        self.sub_socket.bind(f"tcp://{local}:{self.args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    def init_shared_memory(self):
        self.shm_ref = {}

        obs_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * mul(self.obs_shape),
            dtype=np.float64,
        )  # observation-space
        self.sh_obs_batch = LearnerStorage.set_shared_memory(
            self, obs_batch, "obs_batch"
        )

        act_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * 1, dtype=np.float64
        )  # not one-hot, but action index (scalar)
        self.sh_act_batch = LearnerStorage.set_shared_memory(
            self, act_batch, "act_batch"
        )

        rew_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * 1, dtype=np.float64
        )  # scalar
        self.sh_rew_batch = LearnerStorage.set_shared_memory(
            self, rew_batch, "rew_batch"
        )

        logits_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * self.args.action_space,
            dtype=np.float64,
        )  # action-space (logits)
        self.sh_logits_batch = LearnerStorage.set_shared_memory(
            self, logits_batch, "logits_batch"
        )

        is_fir_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * 1, dtype=np.float64
        )  # scalar
        self.sh_is_fir_batch = LearnerStorage.set_shared_memory(
            self, is_fir_batch, "is_fir_batch"
        )

        hx_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * self.args.hidden_size,
            dtype=np.float64,
        )  # hidden-states
        self.sh_hx_batch = LearnerStorage.set_shared_memory(self, hx_batch, "hx_batch")

        cx_batch = np.zeros(
            self.args.seq_len * self.args.batch_size * self.args.hidden_size,
            dtype=np.float64,
        )  # cell-states
        self.sh_cx_batch = LearnerStorage.set_shared_memory(self, cx_batch, "cx_batch")

        self.sh_data_num = mp.Value("i", 0)
        self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

        self._init = True
        return

    @staticmethod
    def set_shared_memory(self, np_array, name):
        assert name in self.dataframe_keyword

        # shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        shm_array = mp.Array(
            "d", len(np_array)
        )  # shm_array: 공유메모리 / shm_array.get_obj(): 공유메모리 공간의 메모리 스페이스 주소

        # setattr(self, f"sh_{name}", np.frombuffer(buffer=shm.buf, dtype=np_array.dtype, count=-1))
        # setattr(self, f"sh_{name}_ref", shm.name)
        # return np_array.shape, np_array.dtype, shm.name, shm.buf # 공유메모리의 (이름, 버퍼)
        # return np.frombuffer(buffer=shm.buf, dtype=np_array.dtype, count=-1)
        return np.frombuffer(buffer=shm_array.get_obj(), dtype=np_array.dtype, count=-1)

    def reset_data_num(self):
        self.sh_data_num.value = 0
        return

    async def shared_memory_chain(self):
        self.proxy_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.retrieve_data_from_worker()),
            asyncio.create_task(self.build_as_batch()),
            asyncio.create_task(self.proxy_q_chain()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def build_as_batch(self):
        while True:
            data = await self.rollout_assembler.pop()
            self.make_batch(data)

            await asyncio.sleep(0.01)

    async def retrieve_data_from_worker(self):
        while True:
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                # with self.mutex():
                await self.rollout_assembler.push(data)

            elif protocol is Protocol.Stat:
                self.stat_list.append(data["mean_stat"])
                if len(self.stat_list) >= self.stat_log_len:
                    mean_stat = self.process_stat()
                    self.log_stat_tensorboard(
                        {"log_len": self.stat_log_len, "mean_stat": mean_stat}
                    )
                    self.stat_list = []
            else:
                assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.01)

    async def proxy_q_chain(self):
        while True:
            # with self.mutex():
            if self.is_sh_ready():
                batch_args = self.get_batch_from_sh_memory()
                await self.proxy_queue.put(batch_args)
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

            await asyncio.sleep(0.01)

    async def put_batch_to_batch_q(self):
        while True:
            # with self.mutex():
            batch_args = await self.proxy_queue.get()
            self.batch_queue.put(batch_args)

            await asyncio.sleep(0.01)

    def process_stat(self):
        mean_stat = {}
        for stat_dict in self.stat_list:
            for k, v in stat_dict.items():
                if not k in mean_stat:
                    mean_stat[k] = [v]
                else:
                    mean_stat[k].append(v)

        mean_stat = {k: np.mean(v) for k, v in mean_stat.items()}
        return mean_stat

    @counted
    def log_stat_tensorboard(self, data):
        len = data["log_len"]
        data_dict = data["mean_stat"]

        for k, v in data_dict.items():
            tag = f"worker/{len}-game-mean-stat-of-{k}"
            # x = self.idx
            x = self.log_stat_tensorboard.calls * len  # global game counts
            y = v
            self.writer.add_scalar(tag, y, x)
            print(f"tag: {tag}, y: {y}, x: {x}")

    def make_batch(self, rollout):
        sq = self.args.seq_len
        bn = self.args.batch_size

        num = self.sh_data_num.value

        sha = mul(self.obs_shape)
        ac = self.args.action_space
        hs = self.args.hidden_size

        if num < bn:
            obs = rollout["obs"]
            act = rollout["act"]
            rew = rollout["rew"]
            logits = rollout["logits"]
            is_fir = rollout["is_fir"]
            hx = rollout["hx"]
            cx = rollout["cx"]

            # 공유메모리에 학습 데이터 적재
            self.sh_obs_batch[sq * num * sha : sq * (num + 1) * sha] = flatten(obs)
            self.sh_act_batch[sq * num : sq * (num + 1)] = flatten(act)
            self.sh_rew_batch[sq * num : sq * (num + 1)] = flatten(rew)
            self.sh_logits_batch[sq * num * ac : sq * (num + 1) * ac] = flatten(logits)
            self.sh_is_fir_batch[sq * num : sq * (num + 1)] = flatten(is_fir)
            self.sh_hx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(hx)
            self.sh_cx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(cx)

            num += 1

    def is_sh_ready(self):
        # sq = self.args.seq_len
        bn = self.args.batch_size

        # assert "batch_num" in self.dataframe_keyword

        # nshape = self.shm_ref["batch_num"][0]
        # ndtype = self.shm_ref["batch_num"][1]
        # nshm = self.shm_ref["batch_num"][2]
        # bshm = self.shm_ref["batch_num"][3]

        # shm_obj = shared_memory.SharedMemory(name=nshm)
        # self.sh_data_num = np.frombuffer(buffer=bshm, dtype=ndtype)
        # self.sh_data_num = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # self.sh_data_num = np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        # if (self.sh_data_num >= self.args.batch_size).item():
        if self.sh_data_num.value >= bn:
            return True
        else:
            return False

    @staticmethod
    def copy_to_ndarray(src):
        # shm_obj = shared_memory.SharedMemory(name=nshm)
        dst = np.empty(src.shape, dtype=src.dtype)
        # src = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # src = np.frombuffer(buffer=bshm, dtype=ndtype)

        # return np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # return np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        np.copyto(dst, src)  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst

    def get_batch_from_sh_memory(self):
        sq = self.args.seq_len
        bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        ac = self.args.action_space

        # (seq, batch, feat)
        sh_obs_bat = LearnerStorage.copy_to_ndarray(self.sh_obs_batch).reshape(
            (sq, bn, *sha)
        )
        sh_act_bat = LearnerStorage.copy_to_ndarray(self.sh_act_batch).reshape(
            (sq, bn, 1)
        )
        sh_rew_bat = LearnerStorage.copy_to_ndarray(self.sh_rew_batch).reshape(
            (sq, bn, 1)
        )
        sh_logits_bat = LearnerStorage.copy_to_ndarray(self.sh_logits_batch).reshape(
            (sq, bn, ac)
        )
        sh_is_fir_bat = LearnerStorage.copy_to_ndarray(self.sh_is_fir_batch).reshape(
            (sq, bn, 1)
        )
        sh_hx_bat = LearnerStorage.copy_to_ndarray(self.sh_hx_batch).reshape(
            (sq, bn, hs)
        )
        sh_cx_bat = LearnerStorage.copy_to_ndarray(self.sh_cx_batch).reshape(
            (sq, bn, hs)
        )

        return (
            to_torch(sh_obs_bat),
            to_torch(sh_act_bat),
            to_torch(sh_rew_bat),
            to_torch(sh_logits_bat),
            to_torch(sh_is_fir_bat),
            to_torch(sh_hx_bat),
            to_torch(sh_cx_bat),
        )
