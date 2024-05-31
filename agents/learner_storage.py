import time
import zmq
import zmq.asyncio
import asyncio

import numpy as np
import multiprocessing as mp

from .storage_module.shared_batch import SMInterFace

from buffers.rollout_assembler import RolloutAssembler
from utils.lock import Mutex
from utils.utils import (
    Protocol,
    mul,
    decode,
    flatten,
    counted,
)


# timer = ExecutionTimer(num_transition=Params.seq_len*1) # LearnerStorage에서 데이터 처리량 (수신) / 부정확한 값이지만 어쩔 수 없음


class LearnerStorage(SMInterFace):
    def __init__(
        self,
        args,
        mutex,
        shm_ref,
        stop_event,
        learner_ip,
        learner_port,
        obs_shape,
        shared_stat_array=None,
        heartbeat=None,
    ):
        super().__init__(shm_ref=shm_ref)

        self.args = args
        self.stop_event = stop_event
        self.obs_shape = obs_shape

        self.mutex: Mutex = mutex

        if shared_stat_array is not None:
            self.np_shared_stat_array: np.ndarray = np.frombuffer(
                buffer=shared_stat_array.get_obj(), dtype=np.float32, count=-1
            )

        self.heartbeat = heartbeat

        self.zeromq_set(learner_ip, learner_port)
        self.get_shared_memory_interface()

    def __del__(self):  # 소멸자
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, learner_ip, learner_port):
        context = zmq.asyncio.Context()

        # learner-storage <-> manager
        self.sub_socket = context.socket(zmq.SUB)  # subscribe batch-data, stat-data
        self.sub_socket.bind(f"tcp://{learner_ip}:{learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def shared_memory_chain(self):
        self.rollout_assembler = RolloutAssembler(self.args, asyncio.Queue(1024))

        tasks = [
            asyncio.create_task(self.retrieve_rollout_from_worker()),
            asyncio.create_task(self.build_as_batch()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_rollout_from_worker(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                await self.rollout_assembler.push(data)

            elif protocol is Protocol.Stat:
                self.log_stat_tensorboard(
                    {"log_len": data["log_len"], "mean_stat": data["mean_stat"]}
                )
            else:
                assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.001)

    async def build_as_batch(self):
        while not self.stop_event.is_set():
            if self.heartbeat is not None:
                self.heartbeat.value = time.time()

            # with timer.timer("learner-storage-throughput", check_throughput=True):
            data = await self.rollout_assembler.pop()
            self.make_batch(data)
            print("rollout is poped !")

            await asyncio.sleep(0.001)

    @counted
    def log_stat_tensorboard(self, data):
        _len = data["log_len"]
        _epi_rew = data["mean_stat"]

        tag = f"worker/{_len}-game-mean-stat-of-epi-rew"
        x = self.log_stat_tensorboard.calls * _len  # global game counts
        y = _epi_rew

        print(f"tag: {tag}, y: {y}, x: {x}")

        # TODO: 좋은 구조는 아님
        if self.np_shared_stat_array is not None:
            assert self.np_shared_stat_array.size == 3

            self.np_shared_stat_array[0] = x  # global game counts
            self.np_shared_stat_array[1] = y  # mean-epi-rew
            self.np_shared_stat_array[2] = 1  # 기록 가능 활성화 (activate)

    def make_batch(self, rollout):
        sq = self.args.seq_len
        # bn = self.args.batch_size

        num = self.sh_data_num.value

        sha = mul(self.obs_shape)
        ac = self.args.action_space
        hs = self.args.hidden_size

        # buf = self.args.buffer_size
        mem_size = int(
            self.sh_obs_batch.shape[0] / (sq * sha)
        )  # TODO: 좋은 코드는 아닌 듯..
        # assert buf == mem_size

        if num < mem_size:
            obs = rollout["obs"]
            act = rollout["act"]
            rew = rollout["rew"]
            logits = rollout["logits"]
            log_prob = rollout["log_prob"]
            is_fir = rollout["is_fir"]
            hx = rollout["hx"]
            cx = rollout["cx"]

            # 공유메모리에 학습 데이터 적재
            self.sh_obs_batch[sq * num * sha : sq * (num + 1) * sha] = flatten(obs)
            self.sh_act_batch[sq * num : sq * (num + 1)] = flatten(act)
            self.sh_rew_batch[sq * num : sq * (num + 1)] = flatten(rew)
            self.sh_logits_batch[sq * num * ac : sq * (num + 1) * ac] = flatten(logits)
            self.sh_log_prob_batch[sq * num : sq * (num + 1)] = flatten(log_prob)
            self.sh_is_fir_batch[sq * num : sq * (num + 1)] = flatten(is_fir)
            self.sh_hx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(hx)
            self.sh_cx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(cx)

            self.sh_data_num.value += 1
