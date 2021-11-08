import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import gym
import numpy as np
import torch
from utils.utils import obs_preprocess
from torch.utils.tensorboard import SummaryWriter



class Worker():
    def __init__(self, args, q_worker, learner, model, rollouts, worker_name):
        self.device = torch.device('cpu')
        self.args = args

        self.q_worker = q_worker # buffer of workers
        self.learner = learner
        self.model = model
        self.rollouts = rollouts # buffer of each single-worker
        self.worker_name = worker_name

        self.writer = SummaryWriter(log_dir=self.args.result_dir)


    def zeromq_settings(self):
        pass

    def publish_exploration_data_to_learner(self):
        pass

    def subscribe_model_from_main_manager(self):
        pass

    
    def log_tensorboard(self):
        if self.num_epi % self.args.log_interval == 0 and self.num_epi != 0:
            self.writer.add_scalar(self.worker_name + '_epi_reward', self.epi_reward, self.num_epi)
            
        self.epi_reward = 0

    def collect_data(self):
        print( 'Build Environment for Worker {}'.format(self.worker_name) )
        
        # init-reset
        done = False 
        self.num_epi, self.epi_reward = 0, 0
        lstm_hidden_state = ( torch.zeros( (1, 1, self.args.hidden_size) ), torch.zeros( (1, 1, self.args.hidden_size) ) ) # (h_s, c_s) / (seq, batch, hidden)
        
        self.env = gym.make(self.args.env)

        while True:
            obs = self.env.reset()
            obs = obs_preprocess(obs, self.args.need_conv)
            
            self.model.load_state_dict( self.learner.model.state_dict() ) # reload learned-model from learner
                                                                          # may need transform from gpu-tensor to cpu-tensor
            # init worker-buffer
            self.rollouts.reset() # init or flush

            for step in range(self.args.time_horizon):
                action, log_prob, next_lstm_hidden_state = self.model.act( obs, lstm_hidden_state )
                next_obs, reward, done, _ = self.env.step( action.item() )
                next_obs = obs_preprocess(next_obs, self.args.need_conv)
                
                self.epi_reward += reward
                reward = np.clip(reward, self.args.reward_clip[0], self.args.reward_clip[1])

                mask = torch.FloatTensor( [ [0.0] if done else [1.0] ] )

                self.rollouts.insert(obs,                                        # (1, c, h, w) or (1, D)
                                     action.view(1, -1),                         # (1, 1) / not one-hot, but action index
                                     torch.from_numpy( np.array( [[reward]] ) ), # (1, 1)
                                     log_prob.view(1, -1),                       # (1, 1)               
                                     mask,                                       # (1, 1)
                                     lstm_hidden_state)                          # (h_s, c_s) / (seq, batch, hidden)
                
                
                obs = next_obs                                   # ( 1, c, h, w )
                lstm_hidden_state = next_lstm_hidden_state       # ( (1, 1, d_h), (1, 1, d_c) )
                
                if done:
                    break
                
            self.num_epi += 1
            
            # squeeze batch dim 
            # (seq, batch, feat) / seq~=time_horizon, batch=1
            self.q_worker.put( (self.rollouts.obs[:], 
                                self.rollouts.actions[:], 
                                self.rollouts.rewards[:],
                                self.rollouts.log_probs[:], 
                                self.rollouts.masks[:],
                                self.rollouts.lstm_hidden_states[:]) )

            self.log_tensorboard()