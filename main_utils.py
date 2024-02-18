import os, sys
import argparse
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

from utils.utils import KillProcesses, SaveErrorLog, Params, WriterClass
from utils.lock import Mutex


mutex = Mutex()


fn_dict = {}


def register(fn):
    fn_dict[fn.__name__] = fn
    return fn


# TODO: 이런 하드코딩 스타일은 바람직하지 않음. 더 좋은 코드 구조로 개선 필요.
DataFrameKeyword = [
    "obs_batch",
    "act_batch",
    "rew_batch",
    "logits_batch",
    "is_fir_batch",
    "hx_batch",
    "cx_batch",
    "batch_num",
]

args = Params
args.device = torch.device(
    f"cuda:{Params.gpu_idx}" if torch.cuda.is_available() else "cpu"
)
args.result_dir = WriterClass.result_dir
args.model_dir = WriterClass.model_dir
print(f"device: {args.device}")

model_dir = Path(args.model_dir)
# 경로가 디렉토리인지 확인
if not model_dir.is_dir():
    # 디렉토리가 아니라면 디렉토리 생성
    model_dir.mkdir(parents=True, exist_ok=True)

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
# learner_model.share_memory() # 공유메모리 사용
learner_model_state_dict = learner_model.cpu().state_dict()


def extract_err(target_dir: str):
    log_dir = os.path.join(os.getcwd(), "logs", target_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return traceback.format_exc(limit=128), log_dir


def worker_run(args, model, worker_name, port, *obs_shape):
    worker = Worker(args, model, worker_name, port, obs_shape)
    worker.collect_rolloutdata()  # collect rollout
    
    
def storage_run(args, mutex, dataframe_keyword, queue, *obs_shape, stat_queue=None):
    storage = LearnerStorage(args, mutex, dataframe_keyword, queue, obs_shape, stat_queue)
    asyncio.run(storage.shared_memory_chain())


def run_learner(args, mutex, learner_model, queue, stat_queue=None):
    learner = Learner(args, mutex, learner_model, queue, stat_queue)
    learning_switcher = {
        "PPO": learner.learning_ppo,
        "IMPALA": learner.learning_impala,
    }
    learning = learning_switcher.get(args.algo, lambda: AssertionError("Should be PPO or IMPALA"))
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
        worker_process = []
        for i in range(args.num_worker):
            print("Build Worker {:d}".format(i))
            worker_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
            worker_model.to(torch.device("cpu"))
            worker_model.load_state_dict(learner_model_state_dict)
            worker_name = "worker_" + str(i)

            w = Process(
                target=worker_run,
                args=(args, worker_model, worker_name, args.worker_port, *obs_shape),
                daemon=True,
            )  # child-processes
            worker_process.append(w)

        for wp in worker_process:
            wp.start()

        for wp in worker_process:
            wp.join()

    except:
        traceback.print_exc(limit=128)
        
        err, log_dir = extract_err("worker")
        SaveErrorLog(err, log_dir)

        for wp in worker_process:
            wp.terminate()
                
        
@register
def learner_sub_process():
    try:
        queue = mp.Queue(1024)
        stat_queue = mp.Queue(64) # 좋은 구조는 아님.
        
        learner_process = []
        s = Process(
            target=storage_run,
            args=(args, mutex, DataFrameKeyword, queue, *obs_shape),
            kwargs={"stat_queue": stat_queue},
            daemon=True,
        )  # child-processes
        learner_process.append(s)

        l = Process(
            target=run_learner,
            args=(args, mutex, learner_model, queue),
            kwargs={"stat_queue": stat_queue},
            daemon=True,
        )  # child-processes
        learner_process.append(l)

        for lp in learner_process:
            lp.start()

        for lp in learner_process:
            lp.join()

    except:
        traceback.print_exc(limit=128)
        
        err, log_dir = extract_err("learner")
        SaveErrorLog(err, log_dir)

        for lp in learner_process:
            lp.terminate()
            
            
if __name__ == "__main__":
    assert len(sys.argv) == 2
    
    func_name = sys.argv[1]
    if func_name in fn_dict:
        fn_dict[func_name]()
    else:
        assert False, f"Wronf func_name: {func_name}"            