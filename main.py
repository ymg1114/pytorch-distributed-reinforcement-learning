import os, sys
import signal
import atexit
import time
import gym
import copy
import torch
import traceback
import asyncio
import multiprocessing as mp

from pathlib import Path
from multiprocessing import Process
from datetime import datetime

from agents.learner import Learner
from agents.worker import Worker
from agents.learner_storage import LearnerStorage
from buffers.manager import Manager

from utils.utils import (
    KillProcesses,
    SaveErrorLog,
    Params,
    result_dir,
    model_dir,
    extract_file_num,
    DataFrameKeyword,
)
from utils.lock import Mutex


fn_dict = {}


def register(fn):
    fn_dict[fn.__name__] = fn
    return fn


class Runner:
    def __init__(self):
        self.args = Params
        self.args.device = torch.device(
            f"cuda:{Params.gpu_idx}" if torch.cuda.is_available() else "cpu"
        )

        # 미리 정해진 경로가 있다면 그것을 사용
        self.args.result_dir = self.args.result_dir or result_dir
        self.args.model_dir = self.args.model_dir or model_dir
        print(f"device: {self.args.device}")

        _model_dir = Path(self.args.model_dir)
        # 경로가 디렉토리인지 확인
        if not _model_dir.is_dir():
            # 디렉토리가 아니라면 디렉토리 생성
            _model_dir.mkdir(parents=True, exist_ok=True)

        # only to get observation, action space
        env = gym.make(self.args.env)
        # env.seed(0)
        self.n_outputs = env.action_space.n
        self.args.action_space = self.n_outputs
        print("Action Space: ", self.n_outputs)
        print("Observation Space: ", env.observation_space.shape)

        # 현재 openai-gym의 "CartPole-v1" 환경만 한정
        assert not self.args.need_conv or len(env.observation_space.shape) <= 1
        M = __import__("networks.models", fromlist=[None]).MlpLSTM
        self.obs_shape = [env.observation_space.shape[0]]
        env.close()

        self.Model = M(
            *self.obs_shape, self.n_outputs, self.args.seq_len, self.args.hidden_size
        )
        self.Model.to(torch.device("cpu"))  # cpu 모델

    def set_model_weight(self, model_dir):
        model_files = list(Path(model_dir).glob(f"{self.args.algo}_*.pt"))

        prev_model_weight = None
        if len(model_files) > 0:
            sorted_files = sorted(model_files, key=extract_file_num)
            if sorted_files:
                prev_model_weight = torch.load(
                    sorted_files[-1],
                    map_location=torch.device("cpu"),  # 가장 최신 학습 모델 로드
                )

        learner_model_state_dict = self.Model.cpu().state_dict()
        if prev_model_weight is not None:
            learner_model_state_dict = {
                k: v.cpu() for k, v in prev_model_weight.state_dict().items()
            }

        return learner_model_state_dict  # cpu 텐서

    @staticmethod
    def extract_err(target_dir: str):
        log_dir = os.path.join(os.getcwd(), "logs", target_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        return traceback.format_exc(limit=128), log_dir

    @staticmethod
    def worker_run(model, worker_name, args, port, *obs_shape, heartbeat=None):
        worker = Worker(args, model, worker_name, port, obs_shape, heartbeat)
        worker.collect_rolloutdata()  # collect rollout

    @staticmethod
    def storage_run(
        args,
        mutex,
        dataframe_keyword,
        queue,
        *obs_shape,
        shared_stat_array=None,
        heartbeat=None,
    ):
        storage = LearnerStorage(
            args,
            mutex,
            dataframe_keyword,
            queue,
            obs_shape,
            shared_stat_array,
            heartbeat,
        )
        asyncio.run(storage.shared_memory_chain())

    @staticmethod
    def run_learner(
        learner_model, args, mutex, queue, shared_stat_array=None, heartbeat=None
    ):
        learner = Learner(
            args, mutex, learner_model, queue, shared_stat_array, heartbeat
        )
        learning_switcher = {
            "PPO": learner.learning_ppo,
            "IMPALA": learner.learning_impala,
        }
        learning = learning_switcher.get(
            args.algo, lambda: AssertionError("Should be PPO or IMPALA")
        )
        learning()

    @register
    def manager_sub_process(self):
        try:
            manager = Manager(self.args, self.args.worker_port, self.obs_shape)
            asyncio.run(manager.data_chain())
        except:
            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("manager")
            SaveErrorLog(err, log_dir)

    @register
    def worker_sub_process(self):
        try:
            learner_model_state_dict = self.set_model_weight(self.args.model_dir)

            for i in range(self.args.num_worker):
                print("Build Worker {:d}".format(i))
                worker_model = copy.deepcopy(self.Model)
                worker_model.load_state_dict(learner_model_state_dict)
                worker_name = "worker_" + str(i)

                heartbeat = mp.Value("d", time.time())

                src_w = {
                    "target": Runner.worker_run,
                    "args": (
                        worker_model,
                        worker_name,
                        self.args,
                        self.args.worker_port,
                        *self.obs_shape,
                    ),
                    "kwargs": {"heartbeat": heartbeat},
                    "heartbeat": heartbeat,
                    "is_model_reload": True,
                }

                w = Process(
                    target=src_w.get("target"),
                    args=src_w.get("args"),
                    kwargs=src_w.get("kwargs"),
                    daemon=True,
                )  # child-processes
                child_process.update({w: src_w})

            for wp in child_process:
                wp.start()

            # for wp in child_process:
            #     wp.join()

        except:
            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("worker")
            SaveErrorLog(err, log_dir)

            # for wp in child_process:
            #     wp.terminate()

    @register
    def learner_sub_process(self):
        try:
            learner_model_state_dict = self.set_model_weight(self.args.model_dir)
            self.Model.load_state_dict(learner_model_state_dict)

            mutex = Mutex()
            queue = mp.Queue(1024)

            # TODO: 좋은 구조는 아님.
            shared_stat_array = mp.Array(
                "d", 3
            )  # [global game counts, mean-epi-rew, activate]

            heartbeat = mp.Value("d", time.time())

            src_s = {
                "target": Runner.storage_run,
                "args": (self.args, mutex, DataFrameKeyword, queue, *self.obs_shape),
                "kwargs": {
                    "shared_stat_array": shared_stat_array,
                    "heartbeat": heartbeat,
                },
                "heartbeat": heartbeat,
            }

            s = Process(
                target=src_s.get("target"),
                args=src_s.get("args"),
                kwargs=src_s.get("kwargs"),
                daemon=True,
            )  # child-processes
            child_process.update({s: src_s})

            heartbeat = mp.Value("d", time.time())

            src_l = {
                "target": Runner.run_learner,
                "args": (self.Model, self.args, mutex, queue),
                "kwargs": {
                    "shared_stat_array": shared_stat_array,
                    "heartbeat": heartbeat,
                },
                "heartbeat": heartbeat,
                "is_model_reload": True,
            }

            l = Process(
                target=src_l.get("target"),
                args=src_l.get("args"),
                kwargs=src_l.get("kwargs"),
                daemon=True,
            )  # child-processes
            child_process.update({l: src_l})

            for lp in child_process:
                lp.start()

            # run_learner(args, mutex, learner_model, queue, shared_stat_array=shared_stat_array)

            # for lp in child_process:
            #     lp.join()

        except:
            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("learner")
            SaveErrorLog(err, log_dir)

            # for lp in child_process:
            #     lp.terminate()

    def start(self):
        # 자식 프로세스 목록 초기화
        global child_process
        child_process = {}

        def _monitor_child_process(restart_delay=60):
            """TODO: 정상 작동하지 않는 듯..
            텐서보드 기록에 버그를 일으킴.
            일단 사용하지 않음."""

            def _restart_process(src, heartbeat):
                # heartbeat 기록 갱신
                heartbeat.value = time.time()

                # src.update({"heartbeat": heartbeat})
                # kwargs = src.get("kwargs")
                # if "heartbeat" in kwargs:
                #     kwargs["heartbeat"] = heartbeat

                is_model_reload = src.get("is_model_reload")
                if is_model_reload is not None and is_model_reload is True:
                    model = copy.deepcopy(self.Model)
                    model.load_state_dict(self.set_model_weight(self.args.model_dir))

                    args = list(src.get("args"))
                    args[0] = model  # TODO: 하드 코딩 규칙. 첫 번째 인덱스가 모델일 것.

                    src.update({"args": tuple(args)})

                new_p = Process(
                    target=src.get("target"),
                    args=src.get("args"),
                    kwargs=src.get("kwargs"),
                    daemon=True,
                )  # child-processes
                new_p.start()
                child_process.update({new_p: src})

                time.sleep(0.5)

            while True:
                for p in list(child_process.keys()):
                    src = child_process.get(p)

                    heartbeat = src.get("heartbeat")
                    assert heartbeat is not None

                    # 자식 프로세스가 죽었거나, 일정 시간 이상 통신이 안된 경우 -> 재시작
                    if not p.is_alive() or (
                        (time.time() - heartbeat.value) > restart_delay
                    ):
                        # 해당 자식 프로세스 종료
                        p.terminate()
                        p.join()
                        child_process.pop(p)
                        assert not p.is_alive(), f"p: {p}"

                        # 해당 자식 프로세스 신규 생성 및 시작
                        _restart_process(src, heartbeat)

                time.sleep(1.0)

        assert len(sys.argv) == 2
        func_name = sys.argv[1]

        if (
            func_name != "manager_sub_process"
        ):  # manager는 관리할 자식 프로세스가 없기 때문.
            assert func_name in ("worker_sub_process", "learner_sub_process")

            # 자식 프로세스 종료 함수
            def terminate_processes(processes):
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    p.join()

            # 종료 시그널 핸들러 설정
            def signal_handler(signum, frame):
                print("Signal received, terminating processes")
                terminate_processes(child_process)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # 프로세스 종료 시 실행될 함수 등록
            atexit.register(terminate_processes, child_process)

        try:
            if func_name in fn_dict:
                fn_dict[func_name](self)
            else:
                assert False, f"Wronf func_name: {func_name}"
            _monitor_child_process()

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc(limit=128)

            for p in child_process:
                p.terminate()

            for p in child_process:
                p.join()

        finally:
            KillProcesses(os.getpid())


if __name__ == "__main__":
    rn = Runner()
    rn.start()
