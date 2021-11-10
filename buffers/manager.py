import torch
import time
import zmq
import numpy as np
import torch.multiprocessing as mp


class Manager():
    def __init__(self, args, WORKER_PORTS, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.device = torch.device('cpu')
        
        self.zeromq_settings( WORKER_PORTS )
        
        self.reset_batch()
 
    def zeromq_settings(self, WORKER_PORTS):
        # manager <-> learner
        context = zmq.Context()
        self.req_socket = context.socket( zmq.REQ ) # send batch-data
        self.req_socket.bind( f"tcp://127.0.0.1:{self.args.learner_port}" )

        # manager <-> worker 
        context = zmq.Context()
        self.rep_socket = context.socket( zmq.REP ) # receive rollout-data
        for port in WORKER_PORTS:
            self.rep_socket.connect( f"tcp://127.0.0.1:{port}" )

    def publish_batchdata_to_learner(self, batch_data):
        self.req_socket.send_pyobj( batch_data )
        
        _ = self.req_socket.redv_pyobj()
        print( 'Send batch-data to learner !' )
        
    def subscribe_rolloutdata_from_workers(self, q_workers):
        rollout_data = self.rep_socket.recv_pyobj()
        q_workers.put( rollout_data )
        
        self.rep_socket.send_pyobj( "RECEIVED_ROLLOUT_DATA" )
        print( 'Received rollout-data from workers !' )
            
    def reset_batch(self):
        self.workers_obs           = torch.zeros(self.args.seq_len+1, 2*self.args.batch_size, self.obs_shape)
        # self.workers_actions       = torch.zeros(3*self.args.seq_len, 1, self.n_outputs) # one-hot
        self.workers_actions       = torch.zeros(self.args.seq_len+1, 2*self.args.batch_size, 1) # not one-hot, but action index
        self.workers_rewards       = torch.zeros(self.args.seq_len+1, 2*self.args.batch_size, 1)
        self.workers_log_probs     = torch.zeros(self.args.seq_len+1, 2*self.args.batch_size, 1)
        self.workers_masks         = torch.zeros(self.args.seq_len+1, 2*self.args.batch_size, 1)
    
        self.workers_hidden_states = torch.zeros(1, 2*self.args.batch_size, self.args.hidden_size)
        self.workers_cell_states   = torch.zeros(1, 2*self.args.batch_size, self.args.hidden_size)

        self.batch_num = 0

    def make_batch(self, q_workers):
        while True:
            self.subscribe_rolloutdata_from_workers( q_workers )
            data = q_workers.get()
            seq_num = 0
            
            while True:
                if seq_num+self.args.seq_len+1 > len(data[0]):
                    diff = self.args.seq_len - (len(data[0]) - seq_num)

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
                    
                    seq_num += self.args.seq_len
                    self.batch_num += 1 
                    
                    break
                
                else:
                    self.workers_obs[:, self.batch_num] = torch.cat( data[0][seq_num: seq_num+self.args.seq_len+1], dim=0 )
                    self.workers_actions[:, self.batch_num] = torch.cat( data[1][seq_num: seq_num+self.args.seq_len+1], dim=0 )
                    self.workers_rewards[:, self.batch_num] = torch.cat( data[2][seq_num: seq_num+self.args.seq_len+1], dim=0 )
                    self.workers_log_probs[:, self.batch_num] = torch.cat( data[3][seq_num: seq_num+self.args.seq_len+1], dim=0 )
                    self.workers_masks[:, self.batch_num] = torch.cat( data[4][seq_num: seq_num+self.args.seq_len+1], dim=0 )

                    lstm_states = data[5][seq_num]
                    self.workers_hidden_states[:, self.batch_num] = lstm_states[0]
                    self.workers_cell_states[:, self.batch_num] = lstm_states[1]

                    seq_num += self.args.seq_len
                    self.batch_num += 1 

            # produce_batch
            if self.batch_num >= self.args.batch_size:
                self.produce_batch()
                self.reset_batch()
        
    def produce_batch(self):
        o, a, r, log_p, mask, h_s, c_s = self.get_batch()

        batch_data = (o.to(self.device), 
                        a.to(self.device), 
                        r.to(self.device), 
                        log_p.to(self.device), 
                        mask.to(self.device), 
                        h_s.to(self.device), 
                        c_s.to(self.device))
        
        self.publish_batchdata_to_learner( batch_data )
        
    def get_batch(self):
        o = self.workers_obs[:, :self.args.batch_size]    # (seq, batch, feat)
        a = self.workers_actions[:, :self.args.batch_size]
        r = self.workers_rewards[:, :self.args.batch_size]
        log_p = self.workers_log_probs[:, :self.args.batch_size]
        mask = self.workers_masks[:, :self.args.batch_size]
        
        h_s = self.workers_hidden_states[:, :self.args.batch_size]
        c_s = self.workers_cell_states[:, :self.args.batch_size]
        
        return o, a, r, log_p, mask, h_s, c_s