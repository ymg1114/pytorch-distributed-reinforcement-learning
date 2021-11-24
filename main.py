import os, sys
import argparse
import gym
import json
import torch
import numpy as np
import torch.multiprocessing as mp

from datetime import datetime
from copy import deepcopy
from types import SimpleNamespace

from agents.learner import Learner
from agents.worker import Worker

from buffers.pub_manager import pub_Manager
from buffers.sub_manager import sub_Manager
from buffers.learner_batch_manager import learner_batch_Manager
from threading import Thread
from utils.utils import ParameterServer, kill_processes

        
def worker_run(args, model, worker_name, port):
    worker = Worker(args, model, worker_name, port)
    worker.collect_rolloutdata() # collect rollout-data (multi-workers)

def pub_manager_run(q_workers, args, *obs_shape):
    req_m = pub_Manager(args, obs_shape)
    req_m.make_batch(q_workers) # make_batch & send batch-data to learner

def sub_manager_run(q_workers, WORKER_PORTS):
    rep_m = sub_Manager(WORKER_PORTS)
    rep_m.sub_rollout_from_workers(q_workers) # received rollout-data from workers

def learner_batch_manager_run(q_batchs, args):
    l_m = learner_batch_Manager(args)
    l_m.sub_batch_from_manager(q_batchs) # received batch-data from manager

def run(q_batchs, args, learner_model, sub_procs):
    [ p.start() for p in sub_procs ]
    learner = Learner(args, learner_model)
    learner.learning(q_batchs)
    [ p.join() for p in sub_procs ]

if __name__ == '__main__':    
    utils = os.path.join(os.getcwd(), "utils", 'parameters.json')
    with open( utils ) as f:
        p = json.load(f)
        p = SimpleNamespace(**p)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=p.target_env)
    
    parser.add_argument('--need-conv', type=bool, default=p.need_conv)
    parser.add_argument('--width', type=int, default=p.w)
    parser.add_argument('--height', type=int, default=p.h)
    parser.add_argument('--is-gray', type=bool, default=p.gray)
    parser.add_argument('--hidden-size', type=int, default=p.hidden_size)
    
    parser.add_argument('--action-repeat', type=bool, default=p.repeat_actions)
    parser.add_argument('--frame-stack', type=bool, default=p.frame_stack)
    
    parser.add_argument('--lr', type=float, default=p.learning_rate)
    
    parser.add_argument('--seq-len', type=int, default=p.unroll_length)

    parser.add_argument('--batch-size', type=int, default=p.batch_size)
    parser.add_argument('--gamma', type=float, default=p.gamma)
    
    parser.add_argument('--time-horizon', type=int, default=p.time_horizon)
    
    parser.add_argument('--cis-hat', type=float, default=p.cis_hat)
    parser.add_argument('--rho-hat', type=float, default=p.rho_hat)
    
    parser.add_argument('--policy-loss-coef', type=float, default=p.policy_loss_coef)
    parser.add_argument('--value-loss-coef', type=float, default=p.value_loss_coef)
    parser.add_argument('--entropy-coef', type=float, default=p.entropy_coef)
    
    parser.add_argument('--max-grad-norm', type=float, default=p.clip_gradient_norm)
    parser.add_argument('--log-interval', type=int, default=p.log_save_interval)
    parser.add_argument('--save-interval', type=int, default=p.model_save_interval)
    parser.add_argument('--reward-clip', type=list, default=p.reward_sacle)
    
    parser.add_argument('--num-worker', type=int, default=p.num_worker)
    
    parser.add_argument('--worker-port', type=int, default=p.worker_port)
    parser.add_argument('--manager-port', type=int, default=p.manager_port)
    parser.add_argument('--learner-port', type=int, default=p.learner_port)
    
    args = parser.parse_args()
    args.device = torch.device('cuda:{p.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    try:
        mp.set_start_method('spawn')
        print("spawn init")
        
        dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
        args.result_dir = os.path.join('results', str(dt_string))
        args.model_dir = os.path.join(args.result_dir, 'models')

        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)

        # Only to get obs, action space
        env = gym.make(args.env)
        n_outputs = env.action_space.n
        print('Action Space: ', n_outputs)
        print('Observation Space: ', env.observation_space.shape)
        
        if args.need_conv or len(env.observation_space.shape) > 1:
            M = __import__("networks.models", fromlist=[None]).ConvLSTM
            obs_shape = [ p.H, p.W, env.observation_space.shape[2] ]
        else:
            M = __import__("networks.models", fromlist=[None]).MlpLSTM
            obs_shape = [ env.observation_space.shape[0] ]
        env.close()
        
        WORKER_PORTS = [ port for port in range(args.worker_port, args.worker_port+args.num_worker) ]
        
        q_workers = mp.Queue(maxsize=3*args.batch_size) # q for multi-worker
        q_batchs = mp.Queue(maxsize=args.batch_size)    # q for learner
        
        learner_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
        learner_model.to( args.device )
        learner_model.share_memory()
        
        sub_procs = []
        rep_m = mp.Process( target=sub_manager_run, args=(q_workers, WORKER_PORTS) ) # sub-processes
        rep_m.daemon = True 
        sub_procs.append(rep_m)

        req_m = mp.Process( target=pub_manager_run, args=(q_workers, args, *obs_shape) ) # sub-processes
        req_m.daemon = True 
        sub_procs.append(req_m)
        
        l_m = mp.Process( target=learner_batch_manager_run, args=(q_batchs, args)  ) # sub-process
        l_m.daemon = True  
        sub_procs.append(l_m)
                
        for i in range( args.num_worker ):
            print('Build Worker {:d}'.format(i))
            worker_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
            worker_model.to( torch.device('cpu') )
            worker_model.share_memory()
            
            worker_name = 'worker_' + str(i)
            w = mp.Process( target=worker_run, args=(args, worker_model, worker_name, WORKER_PORTS[i]) ) # sub-processes
            w.daemon = True 
            sub_procs.append(w)

        run( q_batchs, args, learner_model, sub_procs )
        print(f"Run processes -> num_learner: 1, num_worker: {args.num_worker}")
        
    finally:
        kill_processes()