import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import zmq
import gym
import numpy as np
import torch
from buffers.storage import WorkerRolloutStorage
from utils.utils import obs_preprocess, ParameterServer
from tensorboardX import SummaryWriter


class Worker():
    def __init__(self, args, model, worker_name, port):
        self.device = torch.device('cpu')
        self.args = args

        # self.q_worker = q_worker # buffer of workers
        self.model = model
        self.rollouts = WorkerRolloutStorage() # buffer of each single-worker
        self.worker_name = worker_name
    
        self.zeromq_settings( port )
        
        # self.writer = SummaryWriter(log_dir=self.args.result_dir)

    def zeromq_settings(self, port):
        # worker <-> manager
        context = zmq.Context()
        self.pub_socket = context.socket( zmq.PUB ) # publish rollout-data
        self.pub_socket.bind( f"tcp://127.0.0.1:{port}" )

        context = zmq.Context()
        self.sub_socket = context.socket( zmq.SUB ) # subscribe fresh learner model
        self.sub_socket.connect( f"tcp://127.0.0.1:{self.args.learner_port + 1}" )
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
    def publish_rolloutdata_to_manager(self, rollouts):
        rollout_data = (rollouts.obs[:], 
                        rollouts.actions[:], 
                        rollouts.rewards[:],
                        rollouts.log_probs[:], 
                        rollouts.masks[:],
                        rollouts.lstm_hidden_states[:])
        
        self.pub_socket.send_pyobj( rollout_data )

    def subscribe_model_from_learner(self):
        model_state_dict = self.sub_socket.recv_pyobj()
        return model_state_dict
    
    def log_tensorboard(self):
        if self.num_epi % self.args.log_interval == 0 and self.num_epi != 0:
            self.writer.add_scalar(self.worker_name + '_epi_reward', self.epi_reward, self.num_epi)
        self.epi_reward = 0

    def collect_data(self):
        print( 'Build Environment for {}'.format(self.worker_name) )
        
        # init-reset
        done = False 
        num_epi, epi_reward = 0, 0
        lstm_hidden_state = ( torch.zeros( (1, 1, self.args.hidden_size) ), torch.zeros( (1, 1, self.args.hidden_size) ) ) # (h_s, c_s) / (seq, batch, hidden)
        
        env = gym.make(self.args.env)

        while True:
            obs = env.reset()
            obs = obs_preprocess(obs, self.args.need_conv)
            
            model_state_dict = self.subscribe_model_from_learner()
            if model_state_dict:
                self.model.load_state_dict( model_state_dict ) # reload learned-model from learner
            
            # init worker-buffer
            self.rollouts.reset() # init or flush

            for step in range(self.args.time_horizon):
                action, log_prob, next_lstm_hidden_state = self.model.act( obs, lstm_hidden_state )
                next_obs, reward, done, _ = env.step( action.item() )
                next_obs = obs_preprocess(next_obs, self.args.need_conv)
                
                epi_reward += reward
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
                
            num_epi += 1

            self.publish_rolloutdata_to_manager( self.rollouts )
            
            # # squeeze batch dim 
            # # (seq, batch, feat) / seq~=time_horizon, batch=1
            # self.q_worker.put( (self.rollouts.obs[:], 
            #                     self.rollouts.actions[:], 
            #                     self.rollouts.rewards[:],
            #                     self.rollouts.log_probs[:], 
            #                     self.rollouts.masks[:],
            #                     self.rollouts.lstm_hidden_states[:]) )

            # self.log_tensorboard()