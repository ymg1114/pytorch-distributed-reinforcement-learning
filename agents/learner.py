import os, sys
import torch
import torch.nn as nn
import time
import zmq
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import RMSprop
from tensorboardX import SummaryWriter

class Learner():
    def __init__(self, args, model):
        self.args = args        
        self.q_batch = mp.Queue(maxsize=args.batch_size) # q for learner
        # self.q_batch = q_batch
        self.model = model
        self.model.share_memory() # make other processes can assess
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr)

        self.rho = nn.Parameter( torch.tensor(args.rho_hat, dtype=torch.float), requires_grad=False )
        self.cis = nn.Parameter( torch.tensor(args.cis_hat, dtype=torch.float), requires_grad=False )

        self.zeromq_settings()
        
        # self.writer = SummaryWriter(log_dir=self.args.result_dir) # tensorboard-log

    def zeromq_settings(self):
        context = zmq.Context()
        self.pub_socket = context.socket( zmq.PUB )
        self.pub_socket.bind( f"tcp://127.0.0.1:{self.args.learner_port + 1}" ) # publish fresh learner model

        context = zmq.Context()
        self.sub_socket = context.socket( zmq.SUB )
        self.sub_socket.connect( f"tcp://127.0.0.1:{self.args.learner_port}" ) # subscribe batch-data
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    def publish_model_to_workers(self, model_state_dict):
        self.pub_socket.send_pyobj( model_state_dict )

    def subscribe_batchdata_from_manager(self):
        while True:
            batch_data = self.sub_socket.recv_pyobj()
            self.q_batch.put( batch_data )

    def log_tensorboard(self, loss, detached_losses, idx):
        self.writer.add_scalar('loss', float(loss.item()), idx)
        self.writer.add_scalar('original_value_loss', detached_losses["value"], idx)
        self.writer.add_scalar('original_policy_loss', detached_losses["policy"], idx)
        self.writer.add_scalar('original_entropy', detached_losses["entropy"], idx)
        
    def learning(self):
        idx = 0
        while True:
            # Basically, mini-batch-learning (seq, batch, feat)
            obs, actions, rewards, log_probs, masks, hidden_states, cell_states = self.q_batch.get(block=True)
            #print('Get batch shape: obs: {}, actions: {}, rewards: {}, log_probs: {}, masks: {}, hidden_states: {}, cell_states: {}'.format(obs.shape, actions.shape, rewards.shape, log_probs.shape, masks.shape, hidden_states.shape, cell_states.shape))

            lstm_states = (hidden_states, cell_states) 
            target_log_probs, target_entropy, target_value, lstm_states = self.model( obs,         # (seq+1, batch, c, h, w)
                                                                                      lstm_states, # ( (1, batch, hidden_size), (1, batch, hidden_size) )
                                                                                      masks,       # (seq+1, batch, 1)
                                                                                      actions )    # (seq+1, batch, 1)
            assert rewards.size()[0] == self.args.seq_len+1

            # Copy on the same device at the input tensor
            vtrace = torch.zeros( target_value.size() ).to(self.args.device)

            # Computing importance sampling for truncation levels
            importance_sampling = torch.exp( target_log_probs[:-1] - log_probs[:-1] )  # (seq, batch, 1)
            rho                 = torch.min( self.rho, importance_sampling ) # (seq, batch, 1)
            cis                 = torch.min( self.cis, importance_sampling ) # (seq, batch, 1)

            # Recursive calculus
            # Initialisation : v_{-1}
            # v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
            vtrace[-1] = target_value[-1]  # Bootstrapping

            # Computing the deltas
            delta = rho * ( rewards[:-1] + self.args.gamma * target_value[1:] - target_value[:-1] )

            # Pytorch has no funtion such as tf.scan or theano.scan
            # This disgusting is compulsory for jit as reverse is not supported
            for j in range(self.args.seq_len):  # ex) seq: 10
                i = (self.args.seq_len - 1) - j # 9 ~ 0
                vtrace[i] = (
                    target_value[i]
                    + delta[i]
                    + self.args.gamma * cis[i] * (vtrace[i + 1] - target_value[i + 1])
                )

            # Don't forget to detach !
            # We need to remove the bootstrapping
            vtrace, rho = vtrace.detach(), rho.detach()
      
            v_targets = vtrace.to(self.args.device)
            rhos = rho.to(self.args.device)

            # Losses computation

            # Value loss = l2 target loss -> (v_s - V_w(x_s))**2
            loss_value = (v_targets[:-1] - target_value[:-1]).pow_(2)  
            loss_value = loss_value.sum()

            # Policy loss -> - rho * advantage * log_policy & entropy bonus sum(policy*log_policy)
            # We detach the advantage because we don't compute
            # A = reward + gamma * V_{t+1} - V_t
            # L = - log_prob * A
            # The advantage function reduces variance
            advantage = rewards[:-1] + self.args.gamma * v_targets[1:] - target_value[:-1]
            loss_policy = -rhos * target_log_probs[:-1] * advantage.detach()
            loss_policy = loss_policy.sum()

            # Adding the entropy bonus (much like A3C for instance)
            # The entropy is like a measure of the disorder
            entropy = target_entropy[:-1].sum()

            # Summing all the losses together
            loss = self.args.policy_loss_coef*loss_policy + self.args.value_loss_coef*loss_value - self.args.entropy_coef*entropy

            # These are only used for the statistics
            detached_losses = {
                "policy": loss_policy.detach().cpu(),
                "value": loss_value.detach().cpu(),
                "entropy": entropy.detach().cpu(),
            }
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            print("loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_entropy {:.5f}".format( loss.item(), detached_losses["value"], detached_losses["policy"], detached_losses["entropy"] ))
            self.optimizer.step()
            
            self.publish_model_to_workers( self.model.cpu().state_dict() )
            
            # if (idx % self.args.log_interval == 0):
            #     self.log_tensorboard(loss, detached_losses, idx)

            if (idx % self.args.save_interval == 0):
                torch.save(self.model, os.path.join(self.args.model_dir, f"impala_{idx}.pt"))
            
            idx+= 1