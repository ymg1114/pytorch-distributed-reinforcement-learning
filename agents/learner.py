import os
import zmq
import time
import torch
import asyncio

# import torch.nn.functional as F
# import multiprocessing as mp
import numpy as np

from torch.distributions import Categorical, Uniform
from functools import partial

from utils.lock import Mutex
from utils.utils import (
    Protocol,
    encode,
    make_gpu_batch,
    L_IP,
    ExecutionTimer,
    Params,
    to_torch,
)
from torch.optim import Adam, RMSprop

from .storage_module.shared_batch import SMInterFace
from . import (
    ppo_awrapper,
    v_mpo_awrapper,
    impala_awrapper,
    sac_awrapper,
    sac_continuous_awrapper,
)

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerBase(SMInterFace):
    def __init__(
        self,
        args,
        mutex,
        model,
        shm_ref,
        obs_shape,
        shared_stat_array=None,
        heartbeat=None,
    ):
        super().__init__(shm_ref=shm_ref)
        self.args = args
        self.mutex: Mutex = mutex
        self.obs_shape = obs_shape

        if shared_stat_array is not None:
            self.np_shared_stat_array: np.ndarray = np.frombuffer(
                buffer=shared_stat_array.get_obj(), dtype=np.float32, count=-1
            )

        self.heartbeat = heartbeat

        self.device = self.args.device
        self.model = model.to(self.device)

        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        self.CT = Categorical

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.zeromq_set()
        self.get_shared_memory_interface()
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
        if "value-loss" in detached_losses:
            self.writer.add_scalar(
                "original-value-loss", detached_losses["value-loss"], self.idx
            )

        if "policy-loss" in detached_losses:
            self.writer.add_scalar(
                "original-policy-loss", detached_losses["policy-loss"], self.idx
            )

        if "policy-entropy" in detached_losses:
            self.writer.add_scalar(
                "original-policy-entropy", detached_losses["policy-entropy"], self.idx
            )

        if "ratio" in detached_losses:
            self.writer.add_scalar(
                "min-ratio", detached_losses["ratio"].min(), self.idx
            )
            self.writer.add_scalar(
                "max-ratio", detached_losses["ratio"].max(), self.idx
            )
            self.writer.add_scalar(
                "avg-ratio", detached_losses["ratio"].mean(), self.idx
            )

        if "loss-temperature" in detached_losses:
            self.writer.add_scalar(
                "loss-temperature", detached_losses["loss-temperature"].mean(), self.idx
            )

        if "loss_alpha" in detached_losses:
            self.writer.add_scalar(
                "loss_alpha", detached_losses["loss_alpha"].mean(), self.idx
            )

        if "alpha" in detached_losses:
            self.writer.add_scalar("alpha", detached_losses["alpha"], self.idx)

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

    @staticmethod
    def copy_to_ndarray(src):
        dst = np.empty(src.shape, dtype=src.dtype)
        np.copyto(
            dst, src
        )  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst

    def sample_wrapper(func):
        def _wrap(self, sampling_method):
            sq = self.args.seq_len
            bn = self.args.batch_size
            sha = self.obs_shape
            # hs = self.args.hidden_size
            # ac = self.args.action_space
            buf = self.args.buffer_size

            assert sampling_method in ("on-policy", "off-policy")

            if sampling_method == "off-policy":
                assert buf == int(
                    self.sh_obs_batch.shape[0] / (sq * sha[0])
                )  # TODO: 좋은 코드는 아닌 듯..
                idx = np.random.randint(0, buf, size=bn)
                num_buf = buf
            else:
                assert sampling_method == "on-policy"
                assert bn == int(
                    self.sh_obs_batch.shape[0] / (sq * sha[0])
                )  # TODO: 좋은 코드는 아닌 듯..
                idx = slice(None)  # ":"와 동일
                num_buf = bn

            return func(self, idx, num_buf)

        return _wrap

    @sample_wrapper
    def sample_batch_from_sh_memory(self, idx, num_buf):
        sq = self.args.seq_len
        # bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        ac = self.args.action_space

        _sh_obs_batch = self.sh_obs_batch.reshape((num_buf, sq, *sha))[idx]
        _sh_act_batch = self.sh_act_batch.reshape((num_buf, sq, 1))[idx]
        _sh_rew_batch = self.sh_rew_batch.reshape((num_buf, sq, 1))[idx]
        _sh_logits_batch = self.sh_logits_batch.reshape((num_buf, sq, ac))[idx]
        _sh_log_prob_batch = self.sh_log_prob_batch.reshape((num_buf, sq, 1))[idx]
        _sh_is_fir_batch = self.sh_is_fir_batch.reshape((num_buf, sq, 1))[idx]
        _sh_hx_batch = self.sh_hx_batch.reshape((num_buf, sq, hs))[idx]
        _sh_cx_batch = self.sh_cx_batch.reshape((num_buf, sq, hs))[idx]

        # (batch, seq, feat)
        sh_obs_bat = LearnerBase.copy_to_ndarray(_sh_obs_batch)
        sh_act_bat = LearnerBase.copy_to_ndarray(_sh_act_batch)
        sh_rew_bat = LearnerBase.copy_to_ndarray(_sh_rew_batch)
        sh_logits_bat = LearnerBase.copy_to_ndarray(_sh_logits_batch)
        sh_log_prob_bat = LearnerBase.copy_to_ndarray(_sh_log_prob_batch)
        sh_is_fir_bat = LearnerBase.copy_to_ndarray(_sh_is_fir_batch)
        sh_hx_bat = LearnerBase.copy_to_ndarray(_sh_hx_batch)
        sh_cx_bat = LearnerBase.copy_to_ndarray(_sh_cx_batch)

        return (
            to_torch(sh_obs_bat),
            to_torch(sh_act_bat),
            to_torch(sh_rew_bat),
            to_torch(sh_logits_bat),
            to_torch(sh_log_prob_bat),
            to_torch(sh_is_fir_bat),
            to_torch(sh_hx_bat),
            to_torch(sh_cx_bat),
        )

    @ppo_awrapper(timer=timer)
    def learning_ppo(self): ...

    @v_mpo_awrapper(timer=timer)
    def learning_v_mpo(self): ...

    @impala_awrapper(timer=timer)
    def learning_impala(self): ...

    @sac_awrapper(timer=timer)
    def learning_sac(self): ...

    @sac_continuous_awrapper(timer=timer)
    def learning_sac_continuous(self): ...

    def is_sh_ready(self):
        bn = self.args.batch_size
        val = self.sh_data_num.value
        return True if val >= bn else False

    async def put_batch_to_batch_q(self):
        while True:
            if self.is_sh_ready():
                batch_args = self.sample_batch_from_sh_memory(
                    sampling_method="on-policy"
                )
                await self.batch_queue.put(batch_args)
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화
                print("batch is ready !")

            await asyncio.sleep(0.001)

    async def learning_chain_ppo(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_v_mpo(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_v_mpo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_impala(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_impala()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_sac(self):
        self.batch_buffer = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_sac()),
            asyncio.create_task(self.put_batch_to_buffer_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_sac_continuous(self):
        self.batch_buffer = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_sac_continuous()),
            asyncio.create_task(self.put_batch_to_buffer_q()),
        ]
        await asyncio.gather(*tasks)


class LearnerSinglePPO(LearnerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.model.actor is self.model.critic  # 동일 메모리 주소 참조
        self.actor = self.model.actor.to(self.device)
        self.critic = self.model.critic.to(self.device)


class LearnerSingleIMPALA(LearnerSinglePPO): ...


class LearnerSingleVMPO(LearnerSinglePPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_eta = torch.tensor(
            np.log(self.args.v_mpo_lagrange_multiplier_init), requires_grad=True
        )
        self.log_alpha = torch.tensor(
            np.log(self.args.v_mpo_lagrange_multiplier_init), requires_grad=True
        )

        params = [
            {"params": self.model.parameters()},
            {"params": self.log_eta},
            {"params": self.log_alpha},
        ]
        self.optimizer = RMSprop(
            params, lr=self.args.lr, eps=1e-5
        )  # 옵티마이저 오버라이드

    def get_coef_alpha(self):
        return (
            Uniform(
                torch.tensor(self.args.coef_alpha_below).log(),
                torch.tensor(self.args.coef_alpha_upper).log(),
            )
            .sample()
            .exp()
        )


class LearnerSeperate(LearnerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actor = self.model.actor.to(self.device)
        self.critic = self.model.critic.to(self.device)
        self.target_critic = self.model.critic.to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.args.lr)

        self.target_entropy = (
            self.args.action_space
        )  # 보통 목표 엔트로피 값을 이렇게 설정한다 함..?
        self.log_alpha = torch.tensor(np.log(self.args.alpha), requires_grad=True)
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.args.lr)

    def monitor_flag(func):
        first_return = [False]

        def _wrapper(self, *args, **kwargs):
            nonlocal first_return
            _bool = func(self, *args, **kwargs)

            # 최초 True -> 계속적으로 True 반환
            if _bool is True:
                first_return[0] = True
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

            return first_return[0] or _bool

        return _wrapper

    @monitor_flag
    def is_sh_ready(self) -> bool:
        buf = self.args.buffer_size
        val = self.sh_data_num.value
        return True if val >= buf else False

    async def put_batch_to_buffer_q(self):
        while True:
            if self.is_sh_ready():
                batch_args = self.sample_batch_from_sh_memory(
                    sampling_method="off-policy"
                )
                await self.batch_buffer.put(batch_args)
                print("batch is ready !")

            await asyncio.sleep(0.001)
