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
from torch.multiprocessing import Queue

from agents.learner import Learner
from agents.worker import Worker

from buffers.manager import Manager
from threading import Thread
from utils.utils import ParameterServer


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
    
    learner_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
    learner_model.to(args.device)
    
    manager = Manager(args, WORKER_PORTS, *obs_shape)
    
    learner = Learner(args, learner_model)
    
    processes = []
    l1 = mp.Process(target=learner.learning) # sub-process
    l1.daemon = True  # ensure that the worker exits on process exit

    l2 = mp.Process(target=learner.subscribe_batchdata_from_manager) # sub-process
    l2.daemon = True  

    m1 = mp.Process(target=manager.subscribe_rollout_data_from_workers) # sub-process
    m1.daemon = True

    m2 = mp.Process(target=manager.make_batch) # sub-process
    m2.daemon = True
    
    processes.append(l1)  
    processes.append(l2)
    processes.append(m1)
    processes.append(m2)
    
    for i in range( args.num_worker ):
        print('Build Worker {:d}'.format(i))
        worker_model = M(*obs_shape, n_outputs, args.seq_len, args.hidden_size)
        worker_model.to( torch.device('cpu') )
        
        worker_name = 'worker_' + str(i)
        worker = Worker(args, worker_model, worker_name, WORKER_PORTS[i])
        
        w = mp.Process(target=worker.collect_data) # sub-processes
        w.daemon = True 
        processes.append(w)
        
    print(f"Run processes -> num_learner: 1, num_worker: {args.num_worker}")
    
    [w.start() for w in processes]
    [w.join() for w in processes]



# 하나의 main에는 -> 1개의 러너, 여러개의 액터가 존재 (main 하나에 gpu 하나 할당 하면 되지 않을까 싶음)
# main들 끼리 zeromq 통신이 가능해야함 -> 러너들 각자 독립적으로 학습 -> 그라디언트 발생 -> 모델 업데이트 (여기서 이제 모든 러너에 대해 싱크를 맞춰줘야 함)
# 그런후 (모델 업데이트에 대해 싱크가 맞춰졌기 때문에) 하나의 업데이트된 모델 -> 액터들에게 전송 
# 이러한 논리 전개면, main들을 상위에서 관리해주는 main-manager 이런것도 필요할 듯
# 즉, main -> tmux를 통해 여러개 띄우는 형태로 가야함 (main-manager의 역할)

# gpu 여러개 달린 머신 (하나의 머신) -> gpu 장 수에 맞춰서 -> 러너 개수 할당 (main -> main-learner) / 구조 변경이 필요, 지금은 워커-러너 붙어 있는데, 떼어야 할 듯
# 그리고, 이 한개의 러너 <-> 여러개 워커 통신을 zeromq로 관리해야 할 듯

# 결국에는 트라이앵글 구조 인 듯: (1개)러너-매니저 <-> (여러)러너 <-> 워커들 / 러너-매니저 에서 총 병렬적으로 병합된, 신경망 웨이트 -> 모든 액터들에게 전송