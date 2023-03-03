import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import zmq
import gym
import torch
import numpy as np

from threading import Thread
from utils.utils import Protocol, encode, decode
from utils.utils import obs_preprocess

local = "127.0.0.1"


class Env():
    def __init__(self, args):
        self.args = args
        self._env = gym.make(args.env)

    def reset(self):
        obs, _ = self._env.reset()
        return obs_preprocess(obs, self.args.need_conv)
    
    def step(self, act):
        obs, rew, done, _, _ = self._env_.step(act)
        return obs_preprocess(obs, self.args.need_conv), rew, done
    
    
class Worker():
    def __init__(self, args, model, worker_name, port, obs_shape):
        self.device = torch.device('cpu')
        self.args = args
        self.env = Env(args)
        
        self.model = model
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
                            
    def pub_rollout2(self, **step_data):
        self.pub_socket.send_multipart([*encode(Protocol.Rollout, step_data)]) 
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
        stat.update({'epi_rew': self.epi_rew})
        self.pub_socket.send_multipart([*encode(Protocol.Stat, stat)])
        
    def collect_rolloutdata(self):
        print('Build Environment for {}'.format(self.worker_name))
    
        self.num_epi = 0
        while True:    
            obs = self.env.reset()
            _id = int(np.random.random() * 10000000)
            # print(f"worker_name: {self.worker_name}, obs: {obs}")
            lstm_hx = (torch.zeros((1, 1, self.args.hidden_size)), torch.zeros((1, 1, self.args.hidden_size))) # (h_s, c_s) / (seq, batch, hidden)
            self.epi_rew = 0
            
            # # init worker-buffer
            # self.buffer_reset()    
            
            is_fir = True # first frame
            for _ in range(self.args.time_horizon):
                self.req_model() # every-step
                act, logits, lstm_hx_next = self.model.act(obs, lstm_hx)
                next_obs, rew, done = self.env.step(act.item())
                
                self.epi_rew += rew
                # rew = np.clip(rew, self.args.reward_clip[0], self.args.reward_clip[1])

                step_data = {
                    "obs": obs, # (1, c, h, w) or (1, D)
                    "act": act.view(1, -1), # (1, 1) / not one-hot, but action index
                    "rew": torch.from_numpy(np.array([[rew*self.args.reward_scale]])), # (1, 1)
                    "logits": logits,
                    "is_fir": torch.FloatTensor([[1.0] if is_fir else [0.0]]), # (1, 1),
                    "hx": lstm_hx[0], # (seq, batch, hidden)
                    "cx": lstm_hx[1], # (seq, batch, hidden)
                    "id": _id,
                }
     
                self.pub_rollout2(**step_data)
                
                is_fir = False
                obs = next_obs
                lstm_hx = lstm_hx_next
                
                if done:
                    self.pub_stat()
                    # self.req_model() # every-epi
                    self.epi_rew = 0
                    self.num_epi += 1
                    break

                time.sleep(0.1)
            