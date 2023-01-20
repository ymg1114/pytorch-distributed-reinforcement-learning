import os, sys
import argparse
import gym
import json
import torch
import traceback
import asyncio
import numpy as np
import multiprocessing as mp

from multiprocessing import Process, Semaphore, Pipe, Queue, set_start_method
from datetime import datetime
from copy import deepcopy
from types import SimpleNamespace
from collections import partial

from agents.learner import Learner
from agents.worker import Worker
from agents.learner_storage import LearnerStorage
from buffers.manager import Manager

from threading import Thread
from utils.utils import KillProcesses, SaveErrorLog, Params, SamLock

        
def worker_run(args, model, worker_name, port, *obs_shape):
    worker = Worker(args, model, worker_name, port, obs_shape)
    worker.collect_rolloutdata() # collect rollout
    
    
def manager_run(args, *obs_shape):
    manager = Manager(args, args.worker_port, obs_shape)
    asyncio.run(manager.data_chain())


def storage_run(args, sam_lock, src_conn, *obs_shape):
    storage = LearnerStorage(args, sam_lock, src_conn, obs_shape)
    storage.set_data_to_shared_memory()


def run(args, sam_lock, dst_conn, learner_model, child_procs):
    [p.start() for p in child_procs]
    learner = Learner(args, sam_lock, dst_conn, learner_model)
    learner.learning()              
    [p.join() for p in child_procs]


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    p = Params
    parser.add_argument('--env', type=str, default=p.env)
    
    parser.add_argument('--need-conv', type=bool, default=p.need_conv)
    parser.add_argument('--width', type=int, default=p.w)
    parser.add_argument('--height', type=int, default=p.h)
    parser.add_argument('--is-gray', type=bool, default=p.gray)
    parser.add_argument('--hidden-size', type=int, default=p.hidden_size)
    
    parser.add_argument('--action-repeat', type=bool, default=p.repeat_actions)
    parser.add_argument('--frame-stack', type=bool, default=p.frame_stack)
    
    parser.add_argument('--K-epoch', type=float, default=p.K_epoch)
    parser.add_argument('--lr', type=float, default=p.learning_rate)
    
    parser.add_argument('--seq-len', type=int, default=p.unroll_length)

    parser.add_argument('--batch-size', type=int, default=p.batch_size)
    parser.add_argument('--gamma', type=float, default=p.gamma)
    parser.add_argument('--lmbda', type=float, default=p.lmbda)
    parser.add_argument('--eps-clip', type=float, default=p.eps_clip)
    
    parser.add_argument('--time-horizon', type=int, default=p.time_horizon)
        
    parser.add_argument('--policy-loss-coef', type=float, default=p.policy_loss_coef)
    parser.add_argument('--value-loss-coef', type=float, default=p.value_loss_coef)
    parser.add_argument('--entropy-coef', type=float, default=p.entropy_coef)
    
    parser.add_argument('--max-grad-norm', type=float, default=p.clip_gradient_norm)
    parser.add_argument('--loss-log-interval', type=int, default=p.log_save_interval)
    parser.add_argument('--model-save-interval', type=int, default=p.model_save_interval)
    parser.add_argument('--reward-scale', type=list, default=p.reward_scale)
    
    parser.add_argument('--num-worker', type=int, default=p.num_worker)
    
    parser.add_argument('--worker-port', type=int, default=p.worker_port)
    parser.add_argument('--learner-port', type=int, default=p.learner_port)
    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{p.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {args.device}")
    
    try:
        set_start_method("spawn")
        print("spawn init method run")
        
        dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
        args.result_dir = os.path.join('results', str(dt_string))
        args.model_dir = os.path.join(args.result_dir, 'models')

        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)

        # only to get observation, action space
        env = gym.make(args.env)
        env.seed(0)
        n_outputs = env.action_space.n
        print('Action Space: ', n_outputs)
        print('Observation Space: ', env.observation_space.shape)
        
        # 현재 openai-gym에 한정함
        if args.need_conv and len(env.observation_space.shape) > 1:
            M = __import__("networks.models", fromlist=[None]).ConvLSTM
            obs_shape = [p.H, p.W, env.observation_space.shape[2]]
        else:
            M = __import__("networks.models", fromlist=[None]).MlpLSTM
            obs_shape = [env.observation_space.shape[0]]
        env.close()
        
        src_conn, dst_conn = Pipe()
        # sam = Semaphore(1) # 동시 접근 허용 프로세스 1개
        sam_lock = partial(SamLock, sam=Semaphore(1))
        
        child_procs = []
        # daemonic process is not allowed to create child process
        m = Process(target=manager_run, args=(args, *obs_shape), daemon=True) # child-processes
        child_procs.append(m)
        
        learner_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
        learner_model.to(args.device)
        # learner_model.share_memory() # 공유메모리 사용
        learner_model_state_dict = learner_model.cpu().state_dict()

        for i in range(args.num_worker):
            print('Build Worker {:d}'.format(i))
            worker_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
            worker_model.to(torch.device('cpu'))
            # worker_model.share_memory() # 공유메모리 사용
            worker_model.load_state_dict(learner_model_state_dict)
            
            worker_name = 'worker_' + str(i)
            # daemonic process is not allowed to create child process
            w = Process(target=worker_run, args=(args, worker_model, worker_name, args.worker_port, *obs_shape), daemon=True) # child-processes
            child_procs.append(w)
            
        s = Process(target=storage_run, args=(args, sam_lock, src_conn, *obs_shape), daemon=True) # child-processes
        child_procs.append(s)
        
        run(args, sam_lock, dst_conn, learner_model, child_procs) # main-process
            
    except:
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        err = traceback.format_exc(limit=128)
        SaveErrorLog(err, log_dir)
        
    finally:
        KillProcesses(os.getpid())