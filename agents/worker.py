import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import zmq
import gym
import torch
import numpy as np

from threading import Thread
from utils.utils import encode, decode
from buffers.storage import WorkerRolloutStorage
from utils.utils import obs_preprocess, ParameterServer
from tensorboardX import SummaryWriter

local = "127.0.0.1"

class Worker():
    def __init__(self, args, model, worker_name, port, obs_shape):
        self.device = torch.device('cpu')
        self.args = args
        
        self.model = model
        self.rollouts = WorkerRolloutStorage(args, obs_shape) # buffer of each single-worker
        self.worker_name = worker_name
    
        self.zeromq_set( port )
        
    def zeromq_set(self, port):  
        context = zmq.Context()  
            
        # worker <-> manager
        self.pub_socket = context.socket( zmq.PUB ) 
        self.pub_socket.connect( f"tcp://{local}:{port}" ) # publish rollout-data, stat-data

        # worker <-> learner
        self.sub_socket = context.socket( zmq.SUB ) 
        self.sub_socket.connect( f"tcp://{local}:{self.args.learner_port + 1}" ) # subscribe fresh learner-model
        self.sub_socket.setsockopt( zmq.SUBSCRIBE, b'' )
    
    def sub_model_from_learner(self):
        self.w_t = Thread( target=self.refresh_models )
        self.w_t.daemon = True 
        self.w_t.start()
    
    def refresh_models(self):
        while True:
            try:
                model_state_dict = self.sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
                if model_state_dict:
                    self.model.load_state_dict( model_state_dict )  # reload learned-model from learner
                    # print( f'{self.worker_name}: Received fresh model from learner !' )
            except zmq.Again as e:
                # print("No model-weight received yet")
                pass
                    
            time.sleep(0.01)

    def pub_rollout_to_manager(self):
        rollout_data = (self.rollouts.obs_roll, 
                        self.rollouts.action_roll, 
                        self.rollouts.reward_roll,
                        self.rollouts.log_prob_roll, 
                        self.rollouts.done_roll,
                        self.rollouts.hidden_state_roll,
                        self.rollouts.cell_state_roll
                        )
        
        filter, data = encode('rollout',  rollout_data)
        self.pub_socket.send_multipart( [ filter, data ] ) 
        
    # # NO-BLOCK
    # def req_model_from_learner(self):
    #     try:
    #         model_state_dict = self.sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
    #         if model_state_dict:
    #             self.model.load_state_dict( model_state_dict )  # reload learned-model from learner
    #             # print( f'{self.worker_name}: Received fresh model from learner !' )
    #     except zmq.Again as e:
    #         # print("No model-weight received yet")
    #         pass
        
    # # BLOCK
    # def req_model_from_learner(self):
    #     model_state_dict = self.sub_socket.recv_pyobj()
    #     if model_state_dict:
    #         self.model.load_state_dict( model_state_dict )  # reload learned-model from learner
        
    def pub_stat_to_manager(self):
        stat = {}
        stat.update( {'epi_reward': self.epi_reward} )
        
        filter, data = encode('stat', stat)
        self.pub_socket.send_multipart( [ filter, data ] )
        
    def buffer_reset(self):
        self.rollouts.reset_list()       
        self.rollouts.reset_rolls() 
        self.rollouts.size = 0
        
    def collect_rolloutdata(self):
        print( 'Build Environment for {}'.format(self.worker_name) )
    
        self.num_epi = 0
        env = gym.make(self.args.env)
        
        while True:    
            obs = env.reset()
            obs = obs_preprocess(obs, self.args.need_conv)
            lstm_hidden_state = ( torch.zeros( (1, 1, self.args.hidden_size) ), torch.zeros( (1, 1, self.args.hidden_size) ) ) # (h_s, c_s) / (seq, batch, hidden)
            self.epi_reward = 0
            
            # init worker-buffer
            self.buffer_reset()    
                
            for step in range(self.args.time_horizon):
                # self.req_model_from_learner() # every-step
                
                action, log_prob, next_lstm_hidden_state = self.model.act( obs, lstm_hidden_state )
                next_obs, reward, done, _ = env.step( action.item() )
                next_obs = obs_preprocess(next_obs, self.args.need_conv)
                
                self.epi_reward += reward
                # reward = np.clip(reward, self.args.reward_clip[0], self.args.reward_clip[1])
                _done = torch.FloatTensor( [ [1.0] if done else [0.0] ] )

                self.rollouts.insert(obs,                                                                 # (1, c, h, w) or (1, D)
                                     action.view(1, -1),                                                  # (1, 1) / not one-hot, but action index
                                     torch.from_numpy( np.array( [[ reward*self.args.reward_scale ]] ) ), # (1, 1)
                                     next_obs,                   # (1, c, h, w) or (1, D)
                                     log_prob.view(1, -1),       # (1, 1)               
                                     _done,                      # (1, 1)
                                     lstm_hidden_state)          # (h_s, c_s) / (seq, batch, hidden)
                obs = next_obs                                   # (1, c, h, w) or (1, D)
                lstm_hidden_state = next_lstm_hidden_state       # ( (1, 1, d_h), (1, 1, d_c) )
                
                if self.rollouts.size >= self.args.seq_len or done:
                    self.rollouts.process_rollouts()
                    self.pub_rollout_to_manager()
                    self.buffer_reset() # flush worker-buffer
                    
                    if done:
                        self.pub_stat_to_manager()
                        # self.req_model_from_learner() # every-epi
                        self.epi_reward = 0
                        self.num_epi += 1
                        break
                    
                time.sleep(0.01)
            