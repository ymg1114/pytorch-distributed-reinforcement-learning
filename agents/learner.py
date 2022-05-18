import os, sys
import zmq
import torch
import pickle
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import deque
from utils.utils import encode, decode
from threading import Thread
from torch.optim import RMSprop, Adam
from buffers.batch_buffer import LearnerBatchStorage
from tensorboardX import SummaryWriter

local = "127.0.0.1"

def counted(f):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

class Learner():
    def __init__(self, args, obs_shape, model):
        self.args = args        
        self.model = model.to( args.device )
        # self.model.share_memory() # make other processes can assess
        # self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        
        self.rho = nn.Parameter( torch.tensor(args.rho_hat, dtype=torch.float), requires_grad=False )
        self.cis = nn.Parameter( torch.tensor(args.cis_hat, dtype=torch.float), requires_grad=False )
        
        self.q_workers = mp.Queue(maxsize=args.batch_size) # q for multi-worker-rollout 
        self.batch_buffer = LearnerBatchStorage( args, obs_shape )

        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set()
        self.mean_cal_interval = 30
        self.writer = SummaryWriter(log_dir=self.args.result_dir) # tensorboard-log

    def zeromq_set(self):
        context = zmq.Context()
        
        # learner <-> worker
        self.sub_socket = context.socket( zmq.SUB ) # subscribe rollout-data, stat-data
        self.sub_socket.bind( f"tcp://{local}:{self.args.worker_port}" )
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.pub_socket = context.socket( zmq.PUB )
        self.pub_socket.bind( f"tcp://{local}:{self.args.learner_port}" ) # publish fresh learner-model

    def sub_data_from_workers_Thread(self):
        self.l_t = Thread( target=self.receive_data )
        self.l_t.daemon = True 
        # self.l_t.start()
        
        return self.l_t
        
    def receive_data(self):
        while True:
            filter, data = decode(self.sub_socket.recv_multipart())
            if filter == 'rollout':
                if self.q_workers.qsize() == self.q_workers._maxsize:
                    self.q_workers.get()
                self.q_workers.put( data )
                
            elif filter == 'stat':
                self.stat_list.append( data )
                if len( self.stat_list ) >= self.stat_log_len:
                    mean_stat = self.process_stat()
                    self.log_stat_tensorboard( {"log_len": self.stat_log_len, "mean_stat": mean_stat} )
                    self.stat_list = []
                    
            time.sleep(0.01)

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

    def pub_model_to_workers(self, model_state_dict):        
        self.pub_socket.send_multipart( [encode('model',  model_state_dict)] ) 

    @counted
    def log_stat_tensorboard(self, data):
        len       = data['log_len']
        data_dict = data['mean_stat']
        
        for k, v in data_dict.items():
            tag = f'worker/{len}-game-mean-stat-of-[{k}]'
            # x = self.idx
            x = self.log_stat_tensorboard.calls * len # global game counts
            y = v
            self.writer.add_scalar(tag, y, x)
            print(f'tag: {tag}, y: {y}, x: {x}')
            
    def log_loss_tensorboard(self, loss, detached_losses):
        self.writer.add_scalar('total-loss', float(loss.item()), self.idx)
        self.writer.add_scalar('original-value-loss', detached_losses["value-loss"], self.idx)
        self.writer.add_scalar('original-policy-loss', detached_losses["policy-loss"], self.idx)
        self.writer.add_scalar('original-policy-entropy', detached_losses["policy-entropy"], self.idx)
        
    # # IMPALA
    # def learning(self):
    #     self.idx = 0
    #     while True:
    #         if self.batch_buffer.check_q( self.q_workers ):
    #             # Basically, mini-batch-learning (seq, batch, feat)
    #             batch = self.batch_buffer.roll_to_batch( self.q_workers )
    #             obs, actions, rewards, log_probs, dones, hidden_states, cell_states = batch

    #             # epoch-learning
    #             for _ in range(self.args.K_epoch):

    #                 lstm_states = (hidden_states, cell_states) 
    #                 target_log_probs, target_entropy, target_value, lstm_states = self.model( obs,         # (seq+1, batch, c, h, w)
    #                                                                                         lstm_states, # ( (1, batch, hidden_size), (1, batch, hidden_size) )
    #                                                                                         actions )    # (seq, batch, 1)
    #                 # Copy on the same device at the input tensor
    #                 vtrace = torch.zeros( target_value.size() ).to(self.args.device)

    #                 # masking
    #                 mask = 1 - dones * torch.roll(dones, 1, 0)
    #                 mask_next = torch.roll(mask, -1, 0) + dones*torch.roll(dones, -1, 0) - dones
                    
    #                 value_current = target_value[:-1] * mask     # current
    #                 value_next    = target_value[1:] * mask_next # next
                    
    #                 target_log_probs = target_log_probs * mask # current
    #                 target_entropy = target_entropy * mask     # current
    #                 log_probs = log_probs * mask               # current
                    
    #                 # Recursive calculus
    #                 # Initialisation : v_{-1}
    #                 # v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
    #                 vtrace[-1] = value_next[-1]  # Bootstrapping
                    
    #                 # Computing importance sampling for truncation levels
    #                 importance_sampling = torch.exp( target_log_probs - log_probs )  # (seq, batch, 1)
    #                 rho                 = torch.min( self.rho, importance_sampling ) # (seq, batch, 1)
    #                 cis                 = torch.min( self.cis, importance_sampling ) # (seq, batch, 1)
                    
    #                 # Computing the deltas
    #                 delta = rho * ( rewards + self.args.gamma * value_next - value_current )

    #                 # Pytorch has no funtion such as tf.scan or theano.scan
    #                 # This disgusting is compulsory for jit as reverse is not supported
    #                 for j in range(self.args.seq_len):  # ex) seq: 10
    #                     i = (self.args.seq_len - 1) - j # i: 9 ~ 0
    #                     vtrace[i] = (
    #                         value_current[i]
    #                         + delta[i]
    #                         + self.args.gamma * cis[i] * (vtrace[i + 1] - value_next[i])
    #                     )

    #                 # Don't forget to detach !
    #                 # We need to remove the bootstrapping
    #                 vtrace, rho = vtrace.detach(), rho.detach()
            
    #                 v_targets = vtrace.to(self.args.device)
    #                 rhos = rho.to(self.args.device)

    #                 # Losses computation

    #                 # Value loss = l2 target loss -> (v_s - V_w(x_s))**2
    #                 loss_value = ( v_targets[:-1] - value_current ).pow_(2)  
    #                 loss_value = loss_value.mean()

    #                 # Policy loss -> - rho * advantage * log_policy & entropy bonus sum(policy*log_policy)
    #                 # We detach the advantage because we don't compute
    #                 # A = reward + gamma * V_{t+1} - V_t
    #                 # L = - log_prob * A
    #                 # The advantage function reduces variance
    #                 advantage = rewards + self.args.gamma * v_targets[1:] - value_current
    #                 loss_policy = -rhos * target_log_probs * advantage.detach()
    #                 loss_policy = loss_policy.mean()

    #                 # Adding the entropy bonus (much like A3C for instance)
    #                 # The entropy is like a measure of the disorder
    #                 entropy = target_entropy.mean()

    #                 # Summing all the losses together
    #                 loss = self.args.policy_loss_coef*loss_policy + self.args.value_loss_coef*loss_value - self.args.entropy_coef*entropy

    #                 # These are only used for the statistics
    #                 detached_losses = {
    #                     "policy-loss": loss_policy.detach().cpu(),
    #                     "value-loss": loss_value.detach().cpu(),
    #                     "policy-entropy": entropy.detach().cpu(),
    #                 }
                    
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
    #                 print("loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_policy_entropy {:.5f}".format( loss.item(), detached_losses["value-loss"], detached_losses["policy-loss"], detached_losses["policy-entropy"] ))
    #                 self.optimizer.step()
                
    #             self.pub_model_to_workers( self.model.state_dict() )
                
    #             if (self.idx % self.args.loss_log_interval == 0):
    #                 self.log_loss_tensorboard(loss, detached_losses)

    #             if (self.idx % self.args.model_save_interval == 0):
    #                 torch.save(self.model, os.path.join(self.args.model_dir, f"impala_{self.idx}.pt"))
                
    #             self.idx+= 1
                
    #         time.sleep(0.01)
    
    
    # PPO
    def learning(self):
        self.idx = 0
        
        while True:
            if self.batch_buffer.check_q( self.q_workers ):
                # Basically, mini-batch-learning (seq, batch, feat)
                batch = self.batch_buffer.roll_to_batch( self.q_workers )
                obs, actions, rewards, log_probs, dones, hidden_states, cell_states = batch
  
                # epoch-learning
                for _ in range(self.args.K_epoch):
                    lstm_states = (hidden_states, cell_states) 
                    
                    # on-line model forwarding
                    target_log_probs, target_entropy, target_value, lstm_states = self.model( obs,         # (seq+1, batch, c, h, w)
                                                                                              lstm_states, # ( (1, batch, hidden_size), (1, batch, hidden_size) )
                                                                                              actions )    # (seq, batch, 1)            
                    # masking
                    mask = 1 - dones * torch.roll(dones, 1, 0)
                    mask_next = 1 - dones
                                    
                    value_current = target_value[:-1] * mask     # current
                    value_next    = target_value[1:] * mask_next # next
                    
                    target_log_probs = target_log_probs * mask # current
                    log_probs = log_probs * mask               # current
                    target_entropy = target_entropy * mask     # current
                    
                    # td-target (value-target)
                    # delta for ppo-gae
                    td_target = rewards + self.args.gamma * value_next
                    delta = td_target - value_current
                    delta = delta.cpu().detach().numpy()
                    
                    # ppo-gae (advantage)
                    advantages = []
                    advantage_t = np.zeros(delta.shape[1:]) # Terminal: (seq, batch, d) -> (batch, d)
                    mask_row = mask_next.detach().cpu().numpy()
                    for (delta_row, m_r) in zip(delta[::-1], mask_row[::-1]):
                        advantage_t = delta_row + self.args.gamma * self.args.lmbda * m_r * advantage_t # recursive
                        advantages.append(advantage_t)
                    advantages.reverse()
                    # advantage = torch.stack(advantages, dim=0).to(torch.float)
                    advantage = torch.tensor(advantages, dtype=torch.float).to( self.args.device )

                    ratio = torch.exp(target_log_probs - log_probs)  # a/b == log(exp(a)-exp(b))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantage
                    
                    loss_policy = -torch.min(surr1, surr2).mean()
                    loss_value = F.smooth_l1_loss(value_current, td_target.detach()).mean()  
                    policy_entropy = target_entropy.mean()
                    
                    # Summing all the losses together
                    loss = self.args.policy_loss_coef*loss_policy + self.args.value_loss_coef*loss_value - self.args.entropy_coef*policy_entropy

                    # These are only used for the statistics
                    detached_losses = {
                        "policy-loss": loss_policy.detach().cpu(),
                        "value-loss": loss_value.detach().cpu(),
                        "policy-entropy": policy_entropy.detach().cpu(),
                    }
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    print("loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_policy_entropy {:.5f}".format( loss.item(), detached_losses["value-loss"], detached_losses["policy-loss"], detached_losses["policy-entropy"] ))
                    self.optimizer.step()
                    
                self.pub_model_to_workers( self.model.state_dict() )
                
                if (self.idx % self.args.loss_log_interval == 0):
                    self.log_loss_tensorboard(loss, detached_losses)

                if (self.idx % self.args.model_save_interval == 0):
                    torch.save(self.model, os.path.join(self.args.model_dir, f"impala_{self.idx}.pt"))
                
                self.idx+= 1
                
            time.sleep(0.01)