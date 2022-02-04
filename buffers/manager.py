import torch
import zmq
import numpy as np

from utils.utils import encode, decode
from threading import Thread

local = "127.0.0.1"

class Manager():
    def __init__(self, args, worker_port, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.device = torch.device('cpu')
        
        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set( worker_port )
        self.reset_batch()
 
    def zeromq_set(self, worker_port):
        context = zmq.Context()
        
        # manager <-> worker 
        self.sub_socket = context.socket( zmq.SUB ) # subscribe rollout-data, stat-data
        self.sub_socket.bind( f"tcp://{local}:{worker_port}" )
        self.sub_socket.setsockopt( zmq.SUBSCRIBE, b'' )

        # manager <-> learner
        self.pub_socket = context.socket( zmq.PUB ) # publish batch-data, stat-data
        self.pub_socket.connect( f"tcp://{local}:{self.args.learner_port}"  )

    def pub_batch_to_learner(self, batch):
        _filter, _data = encode('batch', batch)
        self.pub_socket.send_multipart( [ _filter, _data ] )

    def reset_batch(self):
        self.obs_batch          = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
        # self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # not one-hot, but action index (scalar)
        self.reward_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.log_prob_batch     = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.done_batch         = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    
        self.hidden_state_batch = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        self.cell_state_batch   = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        
        self.batch_num = 0
        
    def check_q(self, q_workers):
        if q_workers.qsize() >= self.args.batch_size:
            return True
        else:
            return False
        
    def sub_data_from_workers(self, q_workers):
        self.m_t = Thread( target=self.receive_data, args=(q_workers,) )
        self.m_t.daemon = True 
        self.m_t.start()
    
    def receive_data(self, q_workers):
        while True:
            filter, data = self.sub_socket.recv_multipart()
            filter, data = decode(filter, data)
             
            if filter == 'rollout':
                q_workers.put( data )
                
            elif filter == 'stat':
                self.stat_list.append( data )
                if len( self.stat_list ) >= self.stat_log_len:
                    mean_stat = self.process_stat()
                    _filter, _data = encode('stat', {"log_len": self.stat_log_len, "mean_stat": mean_stat})
                    self.pub_socket.send_multipart( [ _filter, _data ] )
                    self.stat_list = []
    
    def process_stat(self):
        mean_stat = {}
        for stat_dict in self.stat_list:
            for k, v in stat_dict.items():
                if not k in mean_stat:
                    mean_stat[ k ] = [ v ]
                else:
                    mean_stat[ k ].append( v )
                    
        mean_stat = { k: np.mean(v) for k, v in mean_stat.items() }
        return mean_stat
    
    def make_batch(self, q_workers):
        while True:
            if self.check_q( q_workers ):
                for _ in range( self.args.batch_size ):
                    rollout = q_workers.get()

                    obs          = rollout[0]
                    action       = rollout[1]
                    reward       = rollout[2]
                    log_prob     = rollout[3]
                    done         = rollout[4]
                    hidden_state = rollout[5]
                    cell_state   = rollout[6]
                    
                    self.obs_batch[:, self.batch_num] = obs
                    self.action_batch[:, self.batch_num] = action
                    self.reward_batch[:, self.batch_num] = reward 
                    self.log_prob_batch[:, self.batch_num] = log_prob
                    self.done_batch[:, self.batch_num] = done
                    self.hidden_state_batch[:, self.batch_num] = hidden_state
                    self.cell_state_batch[:, self.batch_num] = cell_state
                    
                    self.batch_num += 1
                    
                self.produce_batch()
                self.reset_batch()
                
        # if hasattr(self, "m_t") and self.m_t is not None:
        #     self.m_t.join()
        
    def produce_batch(self):
        o, a, r, log_p, done, h_s, c_s = self.get_batch()

        batch = (o.to(self.device), 
                 a.to(self.device), 
                 r.to(self.device), 
                 log_p.to(self.device), 
                 done.to(self.device), 
                 h_s.to(self.device), 
                 c_s.to(self.device)
                )

        self.pub_batch_to_learner( batch )
        
    def get_batch(self):
        o = self.obs_batch[:, :self.args.batch_size]    # (seq, batch, feat)
        a = self.action_batch[:, :self.args.batch_size]
        r = self.reward_batch[:, :self.args.batch_size]
        log_p = self.log_prob_batch[:, :self.args.batch_size]
        done = self.done_batch[:, :self.args.batch_size]
        
        h_s = self.hidden_state_batch[:, :self.args.batch_size]
        c_s = self.cell_state_batch[:, :self.args.batch_size]
        
        return o, a, r, log_p, done, h_s, c_s