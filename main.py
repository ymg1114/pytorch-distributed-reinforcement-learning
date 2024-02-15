import os
import argparse
import gym

import torch
import traceback
import asyncio
import multiprocessing as mp

from multiprocessing import Process
from datetime import datetime

from agents.learner import Learner
from agents.worker import Worker
from agents.learner_storage import LearnerStorage
from buffers.manager import Manager

from utils.utils import KillProcesses, SaveErrorLog, Params
from utils.lock import Mutex


mutex = Mutex()


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


def worker_run(args, model, worker_name, port, *obs_shape):
    worker = Worker(args, model, worker_name, port, obs_shape)
    worker.collect_rolloutdata()  # collect rollout


def manager_run(args, *obs_shape):
    manager = Manager(args, args.worker_port, obs_shape)
    asyncio.run(manager.data_chain())


def storage_run(args, mutex, dataframe_keyword, queue, *obs_shape):
    storage = LearnerStorage(args, mutex, dataframe_keyword, queue, obs_shape)
    asyncio.run(storage.shared_memory_chain())


def run(args, mutex, learner_model, child_procs, queue):
    [p.start() for p in child_procs]
    learner = Learner(args, mutex, learner_model, queue)
    learner.learning()
    # [p.join() for p in child_procs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    p = Params
    parser.add_argument("--env", type=str, default=p.env)

    parser.add_argument("--need-conv", type=bool, default=p.need_conv)
    parser.add_argument("--width", type=int, default=p.w)
    parser.add_argument("--height", type=int, default=p.h)
    parser.add_argument("--is-gray", type=bool, default=p.gray)
    parser.add_argument("--hidden-size", type=int, default=p.hidden_size)

    parser.add_argument("--K-epoch", type=float, default=p.K_epoch)
    parser.add_argument("--lr", type=float, default=p.learning_rate)

    parser.add_argument("--seq-len", type=int, default=p.unroll_length)

    parser.add_argument("--batch-size", type=int, default=p.batch_size)
    parser.add_argument("--gamma", type=float, default=p.gamma)
    parser.add_argument("--lmbda", type=float, default=p.lmbda)
    parser.add_argument("--eps-clip", type=float, default=p.eps_clip)

    parser.add_argument("--time-horizon", type=int, default=p.time_horizon)

    parser.add_argument("--policy-loss-coef", type=float, default=p.policy_loss_coef)
    parser.add_argument("--value-loss-coef", type=float, default=p.value_loss_coef)
    parser.add_argument("--entropy-coef", type=float, default=p.entropy_coef)

    parser.add_argument("--max-grad-norm", type=float, default=p.clip_gradient_norm)
    parser.add_argument("--loss-log-interval", type=int, default=p.log_save_interval)
    parser.add_argument(
        "--model-save-interval", type=int, default=p.model_save_interval
    )
    parser.add_argument("--reward-scale", type=list, default=p.reward_scale)

    parser.add_argument("--num-worker", type=int, default=p.num_worker)

    parser.add_argument("--worker-port", type=int, default=p.worker_port)
    parser.add_argument("--learner-port", type=int, default=p.learner_port)

    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{p.gpu_idx}" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {args.device}")

    try:
        # set_start_method("spawn")
        # print("spawn init method run")

        dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
        args.result_dir = os.path.join("results", str(dt_string))
        args.model_dir = os.path.join(args.result_dir, "models")

        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)

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

        child_procs = []

        m = Process(
            target=manager_run, args=(args, *obs_shape), daemon=True
        )  # child-processes
        child_procs.append(m)

        learner_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
        learner_model.to(args.device)
        learner_model_state_dict = learner_model.cpu().state_dict()

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
            child_procs.append(w)

        queue = mp.Queue(1024)
        s = Process(
            target=storage_run,
            args=(args, mutex, DataFrameKeyword, queue, *obs_shape),
            daemon=True,
        )  # child-processes
        child_procs.append(s)

        run(args, mutex, learner_model, child_procs, queue)  # main-process

    except:
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        err = traceback.format_exc(limit=128)
        SaveErrorLog(err, log_dir)

    finally:
        KillProcesses(os.getpid())
