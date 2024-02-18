import zmq
import zmq.asyncio
import asyncio

import numpy as np
import multiprocessing as mp

from buffers.rollout_assembler import RolloutAssembler
from utils.utils import Protocol, mul, decode, flatten, to_torch, counted, writer, LS_IP


class LearnerStorage:
    def __init__(self, args, mutex, dataframe_keyword, queue, obs_shape, stat_queue=None):
        self.args = args
        self.dataframe_keyword = dataframe_keyword
        self.obs_shape = obs_shape

        self.stat_list = []
        self.stat_log_len = 20
        self.mutex = mutex
        self._init = False

        self.batch_queue = queue
        self.stat_queue = stat_queue

        self.zeromq_set()
        self.reset_shared_memory()
        # self.writer = writer

    def __del__(self): # 소멸자
        self.sub_socket.close()

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
        self.sub_socket.bind(f"tcp://{LS_IP}:{self.args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    def reset_shared_memory(self):
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
        """멀티프로세싱 환경에서 데이터 복사 없이 공유 메모리를 통해 데이터를 공유함으로써 성능을 개선할 수 있음."""        
        
        assert name in self.dataframe_keyword

        shm_array = mp.Array(
            "d", len(np_array)
        )

        return np.frombuffer(buffer=shm_array.get_obj(), dtype=np_array.dtype, count=-1)

    def reset_data_num(self):
        self.sh_data_num.value = 0
        return

    async def shared_memory_chain(self):
        self.rollout_assembler = RolloutAssembler(self.args, asyncio.Queue(1024))

        tasks = [
            asyncio.create_task(self.retrieve_data_from_worker()),
            asyncio.create_task(self.build_as_batch()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_data_from_worker(self):
        while True:
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                # with self.mutex.lock():
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

    async def build_as_batch(self):
        while True:
            data = await self.rollout_assembler.pop()
            self.make_batch(data)

            await asyncio.sleep(0.01)

    async def put_batch_to_batch_q(self):
        while True:
            # with self.mutex.lock():
            if self.is_sh_ready():
                batch_args = self.get_batch_from_sh_memory()
                self.batch_queue.put(batch_args)
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

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
            x = self.log_stat_tensorboard.calls * len  # global game counts
            y = v
            # self.writer.add_scalar(tag, y, x)
            
            stat_dict = {}
            stat_dict.update(
                {
                    "tag": tag, 
                    "x": x, # global game counts
                    "y": y,
                }
            )                
            self.stat_queue.put(stat_dict)
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

            self.sh_data_num.value += 1

    def is_sh_ready(self):
        bn = self.args.batch_size
        val = self.sh_data_num.value
        return True if val >= bn else False

    @staticmethod
    def copy_to_ndarray(src):
        dst = np.empty(src.shape, dtype=src.dtype)
        np.copyto(dst, src)  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst

    def get_batch_from_sh_memory(self):
        sq = self.args.seq_len
        bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        ac = self.args.action_space

        # (batch, seq, feat)
        sh_obs_bat = LearnerStorage.copy_to_ndarray(self.sh_obs_batch).reshape(
            (bn, sq, *sha)
        )
        sh_act_bat = LearnerStorage.copy_to_ndarray(self.sh_act_batch).reshape(
            (bn, sq, 1)
        )
        sh_rew_bat = LearnerStorage.copy_to_ndarray(self.sh_rew_batch).reshape(
            (bn, sq, 1)
        )
        sh_logits_bat = LearnerStorage.copy_to_ndarray(self.sh_logits_batch).reshape(
            (bn, sq, ac)
        )
        sh_is_fir_bat = LearnerStorage.copy_to_ndarray(self.sh_is_fir_batch).reshape(
            (bn, sq, 1)
        )
        sh_hx_bat = LearnerStorage.copy_to_ndarray(self.sh_hx_batch).reshape(
            (bn, sq, hs)
        )
        sh_cx_bat = LearnerStorage.copy_to_ndarray(self.sh_cx_batch).reshape(
            (bn, sq, hs)
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
