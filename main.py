import os, sys
import argparse
import gym
import json
import torch
import traceback
import numpy as np
import torch.multiprocessing as mp

from datetime import datetime
from copy import deepcopy
from types import SimpleNamespace

from agents.learner import Learner
from agents.worker import Worker
from buffers.manager import Manager

from threading import Thread
from utils.utils import KillProcesses, SaveErrorLog, Params

        
def worker_run(args, model, worker_name, port, *obs_shape):
    worker = Worker(args, model, worker_name, port, obs_shape)
    # worker.model_subscriber()
    worker.collect_rolloutdata() # collect rollout-data (multi-workers)
    
    
def manager_run(q_workers, args, *obs_shape):
    manager = Manager(args, args.worker_port, obs_shape)
    manager.data_subscriber(q_workers) # received rollout-data/stat from workers
    manager.make_batch(q_workers)            # make_batch & send batch-data to learner


def run(q_batchs, args, learner_model, child_procs):
    [p.start() for p in child_procs]
    learner = Learner(args, learner_model)
    learner.data_subscriber(q_batchs) # main-process (sub-thread)
    learner.learning(q_batchs)              # main-process
    [p.join() for p in child_procs]


# # manager가 없는 버전에서 사용했었음
# def run(args, learner_model, child_procs, *obs_shape):
#     learner = Learner(args, obs_shape, learner_model)
#     t1 = learner.data_subscriber() # main-process (sub-thread)
#     child_procs.append(t1)
    
#     [p.start() for p in child_procs]
#     learner.learning()                          # main-process


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
        mp.set_start_method("spawn")
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
        
        if args.need_conv or len(env.observation_space.shape) > 1:
            M = __import__("networks.models", fromlist=[None]).ConvLSTM
            obs_shape = [p.H, p.W, env.observation_space.shape[2]]
        else:
            M = __import__("networks.models", fromlist=[None]).MlpLSTM
            obs_shape = [env.observation_space.shape[0]]
        env.close()
        
        q_workers = mp.Queue(maxsize=args.batch_size) # q for multi-worker (manager)
        q_batchs = mp.Queue(maxsize=1024) # q for learner
        
        child_procs = []
        # daemonic process is not allowed to create child process
        m = mp.Process(target=manager_run, args=(q_workers, args, *obs_shape), daemon=True) # child-processes
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
            w = mp.Process(target=worker_run, args=(args, worker_model, worker_name, args.worker_port, *obs_shape), daemon=True) # child-processes
            child_procs.append(w)
        
        run(q_batchs, args, learner_model, child_procs)
        
        # manager가 없는 버전에서 사용했었음
        # run(args, learner_model, child_procs, *obs_shape)
        print(f"Run processes -> num_learner: 1, num_worker: {args.num_worker}")
    
    except:
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        err = traceback.format_exc(limit=128)
        SaveErrorLog(err, log_dir)
        
    finally:
        KillProcesses(os.getpid())