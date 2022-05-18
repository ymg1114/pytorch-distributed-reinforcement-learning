import torch
import time

from utils import Lock

L = Lock()

class LearnerBatchStorage():
    def __init__(self, args, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.reset_batch()
        
    def reset_batch(self):
        self.obs          = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
        # self.action       = torch.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        self.action       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # not one-hot, but action index (scalar)
        self.reward       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.log_prob     = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
        self.done         = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    
        self.hidden_state = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        self.cell_state   = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        
        self.batch_num = 0
        
    def produce_batch(self):
        o, a, r, log_p, done, h_s, c_s = self.get_batch()

        batch = (o.to(self.args.device), 
                 a.to(self.args.device), 
                 r.to(self.args.device), 
                 log_p.to(self.args.device), 
                 done.to(self.args.device), 
                 h_s.to(self.args.device), 
                 c_s.to(self.args.device)
                )

        return batch

    def get_batch(self):
        o = self.obs    # (seq, batch, feat)
        a = self.action
        r = self.reward
        log_p = self.log_prob
        done = self.done
        
        h_s = self.hidden_state
        c_s = self.cell_state
        
        return o, a, r, log_p, done, h_s, c_s
    
    def check_q(self, q_workers):
        if q_workers.qsize() == self.args.batch_size:
            return True
        else:
            return False
    
    def roll_to_batch(self, q_workers):
        for _ in range( self.args.batch_size ):
            rollout = L.get(q_workers)

            obs          = rollout[0]
            action       = rollout[1]
            reward       = rollout[2]
            log_prob     = rollout[3]
            done         = rollout[4]
            hidden_state = rollout[5]
            cell_state   = rollout[6]
            
            self.obs[:, self.batch_num] = obs
            self.action[:, self.batch_num] = action
            self.reward[:, self.batch_num] = reward 
            self.log_prob[:, self.batch_num] = log_prob
            self.done[:, self.batch_num] = done
            self.hidden_state[:, self.batch_num] = hidden_state
            self.cell_state[:, self.batch_num] = cell_state
            
            self.batch_num += 1
            
        batch = self.produce_batch()        
        self.reset_batch()
        
        return batch