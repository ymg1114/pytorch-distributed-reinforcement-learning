import os, sys
import signal
import atexit
import time
import gym

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


def init():
    args = Params
    args.device = torch.device(
        f"cuda:{Params.gpu_idx}" if torch.cuda.is_available() else "cpu"
    )

    # 미리 정해진 경로가 있다면 그것을 사용
    args.result_dir = args.result_dir or result_dir
    args.model_dir = args.model_dir or model_dir
    print(f"device: {args.device}")

    _model_dir = Path(args.model_dir)
    # 경로가 디렉토리인지 확인
    if not _model_dir.is_dir():
        # 디렉토리가 아니라면 디렉토리 생성
        _model_dir.mkdir(parents=True, exist_ok=True)

    # only to get observation, action space
    env = gym.make(args.env)
    # env.seed(0)
    n_outputs = env.action_space.n
    args.action_space = n_outputs
    print("Action Space: ", n_outputs)
    print("Observation Space: ", env.observation_space.shape)

    # 현재 openai-gym의 "CartPole-v1" 환경만 한정
    assert not args.need_conv or len(env.observation_space.shape) <= 1
    M = __import__("networks.models", fromlist=[None]).MlpLSTM
    obs_shape = [env.observation_space.shape[0]]
    env.close()

    learner_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
    learner_model.to(args.device)
    return args, obs_shape, n_outputs, learner_model, M


def set_model_weight(model_dir):
    model_files = list(Path(model_dir).glob(f"{args.algo}_*.pt"))

    prev_model_weight = None
    if len(model_files) > 0:
        sorted_files = sorted(model_files, key=extract_file_num)
        if sorted_files:
            prev_model_weight = torch.load(
                sorted_files[-1], map_location=torch.device(args.device)
            )

    learner_model_state_dict = learner_model.cpu().state_dict()
    if prev_model_weight is not None:
        learner_model.load_state_dict(prev_model_weight.state_dict())

        learner_model_state_dict = {
            k: v.cpu() for k, v in prev_model_weight.state_dict().items()
        }
    return learner_model_state_dict


def extract_err(target_dir: str):
    log_dir = os.path.join(os.getcwd(), "logs", target_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return traceback.format_exc(limit=128), log_dir


def worker_run(args, model, worker_name, port, *obs_shape, heartbeat=None):
    worker = Worker(args, model, worker_name, port, obs_shape, heartbeat)
    worker.collect_rolloutdata()  # collect rollout


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
        args, mutex, dataframe_keyword, queue, obs_shape, shared_stat_array, heartbeat
    )
    asyncio.run(storage.shared_memory_chain())


def run_learner(
    args, mutex, learner_model, queue, shared_stat_array=None, heartbeat=None
):
    learner = Learner(args, mutex, learner_model, queue, shared_stat_array, heartbeat)
    learning_switcher = {
        "PPO": learner.learning_ppo,
        "IMPALA": learner.learning_impala,
    }
    learning = learning_switcher.get(
        args.algo, lambda: AssertionError("Should be PPO or IMPALA")
    )
    learning()


@register
def manager_sub_process():
    try:
        manager = Manager(args, args.worker_port, obs_shape)
        asyncio.run(manager.data_chain())
    except:
        traceback.print_exc(limit=128)

        err, log_dir = extract_err("manager")
        SaveErrorLog(err, log_dir)


@register
def worker_sub_process():
    try:
        learner_model_state_dict = set_model_weight(args.model_dir)

        for i in range(args.num_worker):
            print("Build Worker {:d}".format(i))
            worker_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
            worker_model.to(torch.device("cpu"))
            worker_model.load_state_dict(learner_model_state_dict)
            worker_name = "worker_" + str(i)

            heartbeat = mp.Value("d", time.time())
            w = Process(
                target=worker_run,
                args=(args, worker_model, worker_name, args.worker_port, *obs_shape),
                kwargs={"heartbeat": heartbeat},
                daemon=True,
            )  # child-processes
            child_process.update({w: heartbeat})

        for wp in child_process:
            wp.start()

        for wp in child_process:
            wp.join()

    except:
        traceback.print_exc(limit=128)

        err, log_dir = extract_err("worker")
        SaveErrorLog(err, log_dir)

        # for wp in child_process:
        #     wp.terminate()


@register
def learner_sub_process():
    try:
        _ = set_model_weight(args.model_dir)

        mutex = Mutex()
        queue = mp.Queue(1024)

        # TODO: 좋은 구조는 아님.
        shared_stat_array = mp.Array(
            "d", 3
        )  # [global game counts, mean-epi-rew, activate]

        heartbeat = mp.Value("d", time.time())
        s = Process(
            target=storage_run,
            args=(args, mutex, DataFrameKeyword, queue, *obs_shape),
            kwargs={"shared_stat_array": shared_stat_array, "heartbeat": heartbeat},
            daemon=True,
        )  # child-processes
        child_process.update({s: heartbeat})

        heartbeat = mp.Value("d", time.time())
        l = Process(
            target=run_learner,
            args=(args, mutex, learner_model, queue),
            kwargs={"shared_stat_array": shared_stat_array, "heartbeat": heartbeat},
            daemon=True,
        )  # child-processes
        child_process.update({l: heartbeat})

        for lp in child_process:
            lp.start()

        # run_learner(args, mutex, learner_model, queue, shared_stat_array=shared_stat_array)

        for lp in child_process:
            lp.join()

    except:
        traceback.print_exc(limit=128)

        err, log_dir = extract_err("learner")
        SaveErrorLog(err, log_dir)

        # for lp in child_process:
        #     lp.terminate()


if __name__ == "__main__":
    args, obs_shape, n_outputs, learner_model, M = init()

    # 자식 프로세스 목록 초기화
    child_process = {}

    def monitor_child_process(restart_delay=20):
        """TODO: 정상 작동하지 않는 듯..
        텐서보드 기록에 버그를 일으킴.
        일단 사용하지 않음."""

        global child_process

        while True:
            for p, heartbeat in child_process.items():
                # 자식 프로세스가 죽었거나, 일정 시간 이상 통신이 안된 경우 -> 재시작
                if not p.is_alive() or (
                    (time.time() - heartbeat.value) > restart_delay
                ):

                    # 모든 자식 프로세스 종료
                    for p in child_process:
                        p.terminate()
                        p.join()
                        assert not p.is_alive(), f"p: {p}"

                    assert (
                        func_name in fn_dict
                    ), f"func_name: {func_name}, fn_dict: {fn_dict}"
                    child_process.clear()  # 자식 프로세스 목록 리셋
                    fn_dict[
                        func_name
                    ]()  # TODO: 자식 프로세스 중 한 개라도 문제가 있으면 전부다 재시작을 강제. 좋은 방법이 아닐지도..?

                    print("All Child Process restarted")
                    break
            time.sleep(1)

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
            fn_dict[func_name]()
        else:
            assert False, f"Wronf func_name: {func_name}"
        # monitor_child_process()

    except Exception as e:
        print(f"error: {e}")
        traceback.print_exc(limit=128)

        for p in child_process:
            p.terminate()

        for p in child_process:
            p.join()

    finally:
        KillProcesses(os.getpid())
