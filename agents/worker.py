import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import zmq
import gym
import asyncio
import torch
import numpy as np

from threading import Thread
from utils.utils import Protocol, encode, decode
from utils.utils import obs_preprocess
from buffers.rollout_buffer import WorkerRolloutStorage

local = "127.0.0.1"

class Worker():
    def __init__(self, args, model, worker_name, port, obs_shape):
        self.device = torch.device('cpu')
        self.args = args
        
        self.model = model
        self.rollouts = WorkerRolloutStorage(args, obs_shape) # buffer single-worker
        self.worker_name = worker_name
    
        self.zeromq_set(port)
        
    def zeromq_set(self, port):  
        context = zmq.Context()  
            
        # worker <-> manager
        self.pub_socket = context.socket(zmq.PUB) 
        self.pub_socket.connect(f"tcp://{local}:{port}") # publish rollout, stat

        self.sub_socket = context.socket(zmq.SUB) 
        self.sub_socket.connect(f"tcp://{local}:{self.args.learner_port+1}") # subscribe model
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
    
    # def model_subscriber(self):
    #     self.w_t = Thread(target=self.refresh_models, daemon=True)
    #     self.w_t.start()

    # def refresh_models(self):
    #     while True:
    #         try:
    #             protocol, data = decode(*self.sub_socket.recv_multipart(flags=zmq.NOBLOCK))
    #             if protocol is Protocol.Model:
    #                 model_state_dict = {k: v.to('cpu') for k, v in data.items()}
    #                 if model_state_dict:
    #                     self.model.load_state_dict(model_state_dict)  # reload learned-model from learner
    #                     # print( f'{self.worker_name}: Received fresh model from learner !' )
    #         except zmq.Again as e:
    #             # print("No model-weight received yet")
    #             pass
                    
    #         time.sleep(0.01)
                    
    def pub_rollout(self):
        rollout_data = (
            self.rollouts.obs_roll, 
            self.rollouts.action_roll, 
            self.rollouts.reward_roll,
            self.rollouts.log_prob_roll, 
            self.rollouts.done_roll,
            self.rollouts.hidden_state_roll,
            self.rollouts.cell_state_roll
            )
        self.pub_socket.send_multipart([*encode(Protocol.Rollout, rollout_data)]) 
        # print(f"worker_name: {self.worker_name} pub rollout to manager!")
        
    # NO-BLOCK
    def req_model(self):
        try:
            protocol, data = decode(*self.sub_socket.recv_multipart(flags=zmq.NOBLOCK))
            if protocol is Protocol.Model:
                model_state_dict = {k: v.to('cpu') for k, v in data.items()}
                if model_state_dict:
                    self.model.load_state_dict(model_state_dict)  # reload learned-model from learner
                    # print( f'{self.worker_name}: Received fresh model from learner !' )
        except zmq.Again as e:
            # print("No model-weight received yet")
            pass
        
    # # BLOCK
    # def req_model(self):
    #     model_state_dict = self.sub_socket.recv_pyobj()
    #     if model_state_dict:
    #         self.model.load_state_dict( model_state_dict )  # reload learned-model from learner
        
    def pub_stat(self):
        stat = {}
        stat.update({'epi_reward': self.epi_reward})
        self.pub_socket.send_multipart([*encode(Protocol.Stat, stat)])
        
    def buffer_reset(self):
        self.rollouts.reset_list()       
        self.rollouts.reset_rolls() 
        self.rollouts.size = 0
        
    def collect_rolloutdata(self):
        print('Build Environment for {}'.format(self.worker_name))
    
        self.num_epi = 0
        env = gym.make(self.args.env)
        
        while True:    
            obs = obs_preprocess(env.reset(), self.args.need_conv)
            # print(f"worker_name: {self.worker_name}, obs: {obs}")
            lstm_hidden_state = (torch.zeros((1, 1, self.args.hidden_size)), torch.zeros((1, 1, self.args.hidden_size))) # (h_s, c_s) / (seq, batch, hidden)
            self.epi_reward = 0
            
            # init worker-buffer
            self.buffer_reset()    
                
            for step in range(self.args.time_horizon):
                self.req_model() # every-step
                action, log_prob, next_lstm_hidden_state = self.model.act(obs, lstm_hidden_state)
                next_obs, reward, done, _ = env.step(action.item())
                next_obs = obs_preprocess(next_obs, self.args.need_conv)
                
                self.epi_reward += reward
                # reward = np.clip(reward, self.args.reward_clip[0], self.args.reward_clip[1])

                self.rollouts.insert(
                    obs, # (1, c, h, w) or (1, D)
                    action.view(1, -1), # (1, 1) / not one-hot, but action index
                    torch.from_numpy(np.array([[reward*self.args.reward_scale]])), # (1, 1)
                    next_obs, # (1, c, h, w) or (1, D)
                    log_prob.view(1, -1), # (1, 1)               
                    torch.FloatTensor([[1.0] if done else [0.0]]), # (1, 1)
                    lstm_hidden_state # (h_s, c_s) / (seq, batch, hidden)
                    )
                
                obs = next_obs # (1, c, h, w) or (1, D)
                lstm_hidden_state = next_lstm_hidden_state # ((1, 1, d_h), (1, 1, d_c))
                
                if self.rollouts.size >= self.args.seq_len or done:
                    self.rollouts.process_rollouts()
                    self.pub_rollout()
                    self.buffer_reset() # flush worker-buffer
                    
                    if done:
                        self.pub_stat()
                        # self.req_model() # every-epi
                        self.epi_reward = 0
                        self.num_epi += 1
                        break
                    
                time.sleep(0.1)
            