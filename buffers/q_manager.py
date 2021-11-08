import torch
import time
import numpy as np
import torch.multiprocessing as mp


class QManager():
    def __init__(self, args, q_worker, q_batch, obs_shape):
        # self.lock = mp.Lock()

        self.q_worker = q_worker
        self.q_batch = q_batch
        self.obs_shape = obs_shape
        
        self.device = torch.device('cpu')
        self.seq_len = args.seq_len       # ex) 10
        self.batch_size = args.batch_size # ex) 64
        
        self.reset_batch()
 
 
    def reset_batch(self):
        self.workers_obs           = torch.zeros(self.seq_len+1, 2*self.batch_size, self.obs_shape)
        # self.workers_actions       = torch.zeros(3*self.seq_len, 1, self.n_outputs) # one-hot
        self.workers_actions       = torch.zeros(self.seq_len+1, 2*self.batch_size, 1) # not one-hot, but action index
        self.workers_rewards       = torch.zeros(self.seq_len+1, 2*self.batch_size, 1)
        self.workers_log_probs     = torch.zeros(self.seq_len+1, 2*self.batch_size, 1)
        self.workers_masks         = torch.zeros(self.seq_len+1, 2*self.batch_size, 1)
    
        self.workers_hidden_states = torch.zeros(1, 2*self.batch_size, 256)
        self.workers_cell_states   = torch.zeros(1, 2*self.batch_size, 256)

        self.batch_num = 0

 
    def make_batch(self):
        while True:
            data = self.q_worker.get(block=True)
            seq_num = 0
            
            while True:
                if seq_num+self.seq_len+1 > len(data[0]):
                    diff = self.seq_len - (len(data[0]) - seq_num)

                    obs_list = data[0][seq_num: ]
                    actions_list = data[1][seq_num: ]
                    rewards_list = data[2][seq_num: ]
                    log_probs_list = data[3][seq_num: ]
                    masks_list = data[4][seq_num: ] 
                    
                    self.workers_obs[:, self.batch_num] = torch.cat( *[ obs_list + [ obs_list[-1] for _ in range(diff+1) ] ], dim=0 )
                    self.workers_actions[:, self.batch_num] = torch.cat( *[ actions_list + [ actions_list[-1] for _ in range(diff+1) ] ], dim=0 )
                    self.workers_rewards[:, self.batch_num] = torch.cat( *[ rewards_list + [ rewards_list[-1] for _ in range(diff+1) ] ], dim=0 )
                    self.workers_log_probs[:, self.batch_num] = torch.cat( *[ log_probs_list + [ log_probs_list[-1] for _ in range(diff+1) ] ], dim=0 )
                    self.workers_masks[:, self.batch_num] = torch.cat( *[ masks_list + [ masks_list[-1] for _ in range(diff+1) ] ], dim=0 )

                    lstm_states = data[5][seq_num]
                    self.workers_hidden_states[:, self.batch_num] = lstm_states[0]
                    self.workers_cell_states[:, self.batch_num] = lstm_states[1]
                    
                    seq_num += self.seq_len
                    self.batch_num += 1 
                    
                    break
                
                else:
                    self.workers_obs[:, self.batch_num] = torch.cat( data[0][seq_num: seq_num+self.seq_len+1], dim=0 )
                    self.workers_actions[:, self.batch_num] = torch.cat( data[1][seq_num: seq_num+self.seq_len+1], dim=0 )
                    self.workers_rewards[:, self.batch_num] = torch.cat( data[2][seq_num: seq_num+self.seq_len+1], dim=0 )
                    self.workers_log_probs[:, self.batch_num] = torch.cat( data[3][seq_num: seq_num+self.seq_len+1], dim=0 )
                    self.workers_masks[:, self.batch_num] = torch.cat( data[4][seq_num: seq_num+self.seq_len+1], dim=0 )

                    lstm_states = data[5][seq_num]
                    self.workers_hidden_states[:, self.batch_num] = lstm_states[0]
                    self.workers_cell_states[:, self.batch_num] = lstm_states[1]

                    seq_num += self.seq_len
                    self.batch_num += 1 

            # produce_batch
            if self.batch_num >= self.batch_size:
                self.produce_batch()
                self.reset_batch()
        
        
    def get_batch(self):
        o = self.workers_obs[:, :self.batch_size]    # (seq, batch, feat)
        a = self.workers_actions[:, :self.batch_size]
        r = self.workers_rewards[:, :self.batch_size]
        log_p = self.workers_log_probs[:, :self.batch_size]
        mask = self.workers_masks[:, :self.batch_size]
        
        h_s = self.workers_hidden_states[:, :self.batch_size]
        c_s = self.workers_cell_states[:, :self.batch_size]
        
        return o, a, r, log_p, mask, h_s, c_s
    
        
    def produce_batch(self):
        o, a, r, log_p, mask, h_s, c_s = self.get_batch()

        # put batch
        self.q_batch.put( (o.to(self.device), 
                           a.to(self.device), 
                           r.to(self.device), 
                           log_p.to(self.device), 
                           mask.to(self.device), 
                           h_s.to(self.device), 
                           c_s.to(self.device)) )

