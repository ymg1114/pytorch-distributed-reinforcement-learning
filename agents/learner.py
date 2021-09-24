import os, sys


import torch
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter

class Learner():
    def __init__(self, args, q_batch, actor_critic):
        self.lr = args.lr
        self.seed = args.seed
        self.device = args.device
        self.model_dir = args.model_dir
        self.result_dir = args.result_dir
        self.save_interval = args.save_interval
        
        self.gamma = args.gamma
        self.cis_hat = args.cis_hat
        self.rho_hat = args.rho_hat
        
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm
        self.policy_loss_coef = args.policy_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        
        self.q_batch = q_batch
        self.actor_critic = actor_critic
        self.optimizer = RMSprop(self.actor_critic.parameters(), lr=self.lr)
        
        self.actor_critic.share_memory() # make other processes can assess

        self.rho = nn.Parameter( torch.tensor(args.rho_hat, dtype=torch.float), requires_grad=False )
        self.cis = nn.Parameter( torch.tensor(args.cis_hat, dtype=torch.float), requires_grad=False )

        self.writer = SummaryWriter(log_dir=self.result_dir) # tensorboard-log
        
        
    def zeromq_settings(self):
        pass

    def publish_model_to_main_manager(self):
        pass

    def subscribe_exploration_data_from_workers(self):
        pass
        
        
    def learning(self):
        torch.manual_seed(self.seed) # seed

        idx = 0
        while True:
            # Basically, (seq, batch, feat)
            obs, actions, rewards, log_probs, masks, hidden_states, cell_states = self.q_batch.get(block=True)
            #print('Get batch shape: obs: {}, actions: {}, rewards: {}, log_probs: {}, masks: {}, hidden_states: {}, cell_states: {}'.format(obs.shape, actions.shape, rewards.shape, log_probs.shape, masks.shape, hidden_states.shape, cell_states.shape))

            lstm_states = (hidden_states, cell_states) 
            target_log_probs, target_entropy, target_value, lstm_states = self.actor_critic( obs,         # (seq, batch, c, h, w)
                                                                                             lstm_states, # ( (1, batch, 256), (1, batch, 256) )
                                                                                             masks,       # (seq, batch, 1)
                                                                                             actions )    # (seq, batch, 1)
            assert rewards.size()[0] == self.seq_len

            # Copy on the same device at the input tensor
            vtrace = torch.zeros( target_value.size() ).to(self.device)

            # Computing importance sampling for truncation levels
            importance_sampling = torch.exp( target_log_probs - log_probs )  # (seq, batch, 1)
            rho                 = torch.min( self.rho, importance_sampling ) # (seq, batch, 1)
            cis                 = torch.min( self.cis, importance_sampling ) # (seq, batch, 1)

            # Recursive calculus
            # Initialisation : v_{-1}
            # v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
            vtrace[-1] = target_value[-1]  # Bootstrapping

            # Computing the deltas
            delta = rho * ( rewards + self.gamma * target_value[1:] - target_value[:-1] )

            # Pytorch has no funtion such as tf.scan or theano.scan
            # This disgusting is compulsory for jit as reverse is not supported
            for j in range(self.seq_len):
                i = (self.seq_len - 1) - j
                vtrace[i] = (
                    target_value[i]
                    + delta[i]
                    + self.gamma * cis[i] * (vtrace[i + 1] - target_value[i + 1])
                )

            # Don't forget to detach !
            # We need to remove the bootstrapping
            vtrace, rho = vtrace.detach(), rho.detach()
      
            v_targets = vtrace.to(self.device)
            rhos = rho.to(self.device)

            # Losses computation

            # Value loss = l2 target loss -> (v_s - V_w(x_s))**2
            loss_value = (v_targets - target_value).pow_(
                2
            )  # No need to remove bootstrap as diff equals zero
            loss_value = loss_value.sum()

            # Policy loss -> - rho * advantage * log_policy & entropy bonus sum(policy*log_policy)
            # We detach the advantage because we don't compute
            # A = reward + gamma * V_{t+1} - V_t
            # L = - log_prob * A
            # The advantage function reduces variance
            advantage = rewards + self.gamma * v_targets[1:] - target_value[:-1]
            loss_policy = -rhos * target_log_probs[:-1] * advantage.detach()
            loss_policy = loss_policy.sum()

            # Adding the entropy bonus (much like A3C for instance)
            # The entropy is like a measure of the disorder
            entropy = target_entropy[:-1].sum()

            # Summing all the losses together
            loss = self.policy_loss_coef*loss_policy + self.value_loss_coef*loss_value - self.entropy_coef*entropy

            # These are only used for the statistics
            detached_losses = {
                "policy": loss_policy.detach().cpu(),
                "value": loss_value.detach().cpu(),
                "entropy": entropy.detach().cpu(),
            }

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            print("loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_entropy {:.5f}".format( loss.item(), detached_losses["value"], detached_losses["policy"], detached_losses["entropy"] ))
            self.optimizer.step()
            
            self.writer.add_scalar('loss', float(loss.item()), idx)

            if (idx % self.save_interval == 0):
                torch.save(self.actor_critic, os.path.join(self.model_dir, f"impala_{idx}.pt"))
            idx+= 1
        
        
        
    # def learning(self):
    #     torch.manual_seed(self.seed) # seed
        
    #     cis_hat = torch.Tensor( [self.cis_hat]*self.seq_len*self.batch_size ).view(self.seq_len, self.batch_size, 1).to(self.device) # (seq, batch, 1)
    #     rho_hat = torch.Tensor( [self.rho_hat]*self.seq_len*self.batch_size ).view(self.seq_len, self.batch_size, 1).to(self.device) # (seq, batch, 1)

    #     idx = 0
    #     while True:
    #         # Basically, (seq, batch, feat)
    #         obs, actions, rewards, log_probs, masks, hidden_states, cell_states = self.q_batch.get(block=True)
    #         #print('Get batch shape: obs: {}, actions: {}, rewards: {}, log_probs: {}, masks: {}, hidden_states: {}, cell_states: {}'.format(obs.shape, actions.shape, rewards.shape, log_probs.shape, masks.shape, hidden_states.shape, cell_states.shape))

    #         lstm_states = (hidden_states, cell_states) 
    #         target_log_probs, target_entropy, target_value, lstm_states = self.actor_critic( obs,         # (seq, batch, c, h, w)
    #                                                                                          lstm_states, # ( (1, batch, 256), (1, batch, 256) )
    #                                                                                          masks,       # (seq, batch, 1)
    #                                                                                          actions )    # (seq, batch, 1)
    #         importance_sampling = torch.exp( target_log_probs - log_probs ) # (seq, batch, 1)
            
    #         cis = torch.min(cis_hat, importance_sampling) ) # (seq, batch, 1)
    #         rho = torch.min(rho_hat, importance_sampling) ) # (seq, batch, 1)
      
    #         policy_loss = 0
    #         value_loss = 0
    #         entropy_loss = 0

    #         # Initialisation : v_{-1}
    #         # v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
    #         vtrace = torch.zeros( (self.seq_len, self.batch_size, 1) ).to(self.device)
    #         vtrace[-1] = target_value[-1]  # Bootstrapping
            
    #         """
    #         v-trace targets are computed recursively
    #         v_s = V(x_s) + delta_sV + gamma*c_s(v_{s+1} - V(x_{s+1}))
    #         vtrace: n-step v-trace target (n -> seq_len)
    #         """
    #         for i in reversed( range(self.seq_len-1) ): # seq_len-1 -> seq_len-2 -> ... -> 0
    #             delta_s = rho[i] * (rewards[i] + self.gamma * target_value[i+1]-target_value[i])
    #             advantages = rho[i] * (rewards[i] + self.gamma * vtrace[i+1] - target_value[i])
    #             vtrace[i] = target_value[i] + delta_s + self.gamma * cis[i] * (vtrace[i+1]-target_value[i+1])

    #             policy_loss += target_log_probs[i]*advantages.detach()

    #         policy_loss = policy_loss.sum()
    #         value_loss = torch.sum( ( vtrace[:-1].detach() - target_value[:-1] ).pow_(2) )
    #         entropy_loss = torch.sum( target_entropy )
            
    #         total_loss = self.policy_loss_coef*policy_loss + self.value_loss_coef*value_loss - self.entropy_coef*entropy_loss

    #         self.optimizer.zero_grad()
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
    #         print("total_loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_entropy_loss {:.5f}".format( total_loss.item(), value_loss.item(), policy_loss.item(), entropy_loss.item() ))
    #         self.optimizer.step()
            
    #         self.writer.add_scalar('total_loss', float(total_loss.item()), idx)

    #         if (idx % self.save_interval == 0):
    #             torch.save(self.actor_critic, os.path.join(self.model_dir, f"impala_{idx}.pt"))
    #         idx+= 1