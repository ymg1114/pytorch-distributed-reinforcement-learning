import torch
import zmq
import numpy as np

from threading import Thread

local = "127.0.0.1"

class Manager():
    def __init__(self, args, worker_port, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.device = torch.device('cpu')
        
        self.zeromq_set( worker_port )
        self.reset_batch()
 
    def zeromq_set(self, worker_port):
        # manager <-> worker 
        context = zmq.Context()
        self.sub_socket = context.socket( zmq.SUB ) # subscribe rollout-data
        self.sub_socket.bind( f"tcp://{local}:{worker_port}" )
        self.sub_socket.setsockopt( zmq.SUBSCRIBE, b'' )

        # manager <-> learner
        context = zmq.Context()
        self.pub_socket = context.socket( zmq.PUB ) # publish batch-data
        self.pub_socket.connect( f"tcp://{local}:{self.args.learner_port}"  )

    def pub_batch_to_learner(self, batch):
        self.pub_socket.send_pyobj( batch )
        # print( 'Send batch-data to learner !' )

    def reset_batch(self):
        self.obs_batch          = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
        # self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # not one-hot, but action index (scalar)
        self.reward_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.log_prob_batch     = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.mask_batch         = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    
        self.hidden_state_batch = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        self.cell_state_batch   = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        
        self.batch_num = 0
        
    def check_q(self, q_workers):
        if q_workers.qsize() >= self.args.batch_size:
            return True
        else:
            return False
        
    def sub_rollout_from_workers(self, q_workers):
        self.m_t = Thread( target=self.receive_rollout, args=(q_workers,) )
        self.m_t.daemon = True 
        self.m_t.start()
    
    def receive_rollout(self, q_workers):
        while True:
            rollout = self.sub_socket.recv_pyobj()
            q_workers.put( rollout )
            # print( 'Received rollout-data from workers !' )
                    
    def make_batch(self, q_workers):
        while True:
            if self.check_q( q_workers ):
                for _ in range( self.args.batch_size ):
                    rollout = q_workers.get()

                    obs          = rollout[0]
                    action       = rollout[1]
                    reward       = rollout[2]
                    log_prob     = rollout[3]
                    mask         = rollout[4]
                    hidden_state = rollout[5]
                    cell_state   = rollout[6]
                    
                    self.obs_batch[:, self.batch_num] = obs
                    self.action_batch[:, self.batch_num] = action
                    self.reward_batch[:, self.batch_num] = reward 
                    self.log_prob_batch[:, self.batch_num] = log_prob
                    self.mask_batch[:, self.batch_num] = mask
                    self.hidden_state_batch[:, self.batch_num] = hidden_state
                    self.cell_state_batch[:, self.batch_num] = cell_state
                    
                    self.batch_num += 1
                    
                self.produce_batch()
                self.reset_batch()
                
        # if hasattr(self, "m_t") and self.m_t is not None:
        #     self.m_t.join()
        
    def produce_batch(self):
        o, a, r, log_p, mask, h_s, c_s = self.get_batch()

        batch = (o.to(self.device), 
                 a.to(self.device), 
                 r.to(self.device), 
                 log_p.to(self.device), 
                 mask.to(self.device), 
                 h_s.to(self.device), 
                 c_s.to(self.device)
                )

        self.pub_batch_to_learner( batch )
        
    def get_batch(self):
        o = self.obs_batch[:, :self.args.batch_size]    # (seq, batch, feat)
        a = self.action_batch[:, :self.args.batch_size]
        r = self.reward_batch[:, :self.args.batch_size]
        log_p = self.log_prob_batch[:, :self.args.batch_size]
        mask = self.mask_batch[:, :self.args.batch_size]
        
        h_s = self.hidden_state_batch[:, :self.args.batch_size]
        c_s = self.cell_state_batch[:, :self.args.batch_size]
        
        return o, a, r, log_p, mask, h_s, c_s