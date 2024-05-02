import zmq.asyncio

import numpy as np
import multiprocessing as mp

from utils.utils import mul, DataFrameKeyword


def set_shared_memory(shm_ref, np_array, name):
    """멀티프로세싱 환경에서 데이터 복사 없이 공유 메모리를 통해 데이터를 공유함으로써 성능을 개선할 수 있음."""

    assert name in DataFrameKeyword

    shm_ref.update(
        {name: (mp.Array("f", len(np_array)), np.float32)}
    )  # {키워드: (공유메모리, 타입), ... }


def setup_shared_memory(args, obs_shape, batch_size):
    shm_ref = {}  # Learner / LearnerStorage 에서 공유할 메모리 주소를 담음

    obs_batch = np.zeros(
        args.seq_len * batch_size * mul(obs_shape),
        dtype=np.float32,
    )  # observation-space
    set_shared_memory(shm_ref, obs_batch, "obs_batch")

    act_batch = np.zeros(
        args.seq_len * batch_size * 1, dtype=np.float32
    )  # not one-hot, but action index (scalar)
    set_shared_memory(shm_ref, act_batch, "act_batch")

    rew_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    set_shared_memory(shm_ref, rew_batch, "rew_batch")

    # logits_batch = np.zeros(
    #     args.seq_len * batch_size * args.action_space,
    #     dtype=np.float32,
    # )  # action-space (logits)
    # set_shared_memory(shm_ref, logits_batch, "logits_batch")

    log_prob_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    set_shared_memory(shm_ref, log_prob_batch, "log_prob_batch")

    is_fir_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    set_shared_memory(shm_ref, is_fir_batch, "is_fir_batch")

    hx_batch = np.zeros(
        args.seq_len * batch_size * args.hidden_size,
        dtype=np.float32,
    )  # hidden-states
    set_shared_memory(shm_ref, hx_batch, "hx_batch")

    cx_batch = np.zeros(
        args.seq_len * batch_size * args.hidden_size,
        dtype=np.float32,
    )  # cell-states
    set_shared_memory(shm_ref, cx_batch, "cx_batch")

    # 공유메모리 저장 인덱스
    sh_data_num = mp.Value("i", 0)
    sh_data_num.value = 0  # 초기화
    shm_ref.update({"batch_index": sh_data_num})
    return shm_ref


def reset_shared_on_policy_memory(args, obs_shape):
    return setup_shared_memory(args, obs_shape, batch_size=args.batch_size)


def reset_shared_buffer_memory(args, obs_shape):
    return setup_shared_memory(args, obs_shape, batch_size=args.buffer_size)


class SMInterFace:
    def __init__(self, shm_ref):
        self.shm_ref = shm_ref

    def get_shared_memory(self, name: str):
        assert hasattr(self, "shm_ref") and name in self.shm_ref

        shm_memory_tuple = self.shm_ref.get(name)
        assert shm_memory_tuple is not None

        shm_array = shm_memory_tuple[0]
        dtype = shm_memory_tuple[1]

        return np.frombuffer(buffer=shm_array.get_obj(), dtype=dtype, count=-1)

    def get_shared_memory_interface(self):
        assert hasattr(self, "shm_ref")

        self.sh_obs_batch = self.get_shared_memory("obs_batch")
        self.sh_act_batch = self.get_shared_memory("act_batch")
        self.sh_rew_batch = self.get_shared_memory("rew_batch")
        # self.sh_logits_batch = self.get_shared_memory("logits_batch")
        self.sh_log_prob_batch = self.get_shared_memory("log_prob_batch")
        self.sh_is_fir_batch = self.get_shared_memory("is_fir_batch")
        self.sh_hx_batch = self.get_shared_memory("hx_batch")
        self.sh_cx_batch = self.get_shared_memory("cx_batch")

        self.sh_data_num = self.shm_ref.get("batch_index")

        self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

    def reset_data_num(self):
        self.sh_data_num.value = 0
