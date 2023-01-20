import torch

class WorkerRolloutStorage():
    def __init__(self, args, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.device = torch.device('cpu')
        self.reset_list()  # init   
        self.reset_rolls() # init
        
    def reset_list(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = None
        self.log_probs = []
        self.dones = []
        self.lstm_hidden_states = []
        self.size = 0
        
    def reset_rolls(self):
        self.obs_roll = torch.zeros(self.args.seq_len+1, *self.obs_shape)
        # self.action_roll = torch.zeros(3*self.args.seq_len, self.n_outputs) # one-hot
        self.action_roll = torch.zeros(self.args.seq_len, 1) # not one-hot, but action index (scalar)
        self.reward_roll = torch.zeros(self.args.seq_len, 1) # scalar
        self.log_prob_roll = torch.zeros(self.args.seq_len, 1) # scalar
        self.done_roll = torch.zeros(self.args.seq_len, 1) # scalar
    
        self.hidden_state_roll = torch.zeros(1, self.args.hidden_size)
        self.cell_state_roll   = torch.zeros(1, self.args.hidden_size)
        
    def insert(self, obs, action, reward, next_obs, log_prob, done, lstm_hidden_state):
        self.obs.append(obs.to(self.device)) # (1, c, h, w) or (1, D)
        self.actions.append(action.to(self.device)) # (1, 1) / not one-hot, but action index
        self.rewards.append(reward.to(self.device)) # (1, 1)
        self.next_obs = next_obs # (1, c, h, w) or (1, D)
        self.log_probs.append(log_prob.to(self.device)) # (1, 1)
        self.dones.append(done.to(self.device)) # (1, 1)
        
        lstm_hidden_state = (lstm_hidden_state[0].to(self.device), lstm_hidden_state[1].to(self.device)) 
        self.lstm_hidden_states.append(lstm_hidden_state) # ((1, 1, d_h), (1, 1, d_c))
        
        self.size += 1
         
    def process_rollouts(self):
        if self.size == self.args.seq_len:
            self.obs_roll[:] = torch.cat(self.obs + [self.next_obs], dim=0)
            self.action_roll[:] = torch.cat(self.actions, dim=0)
            self.reward_roll[:] = torch.cat(self.rewards, dim=0)
            self.log_prob_roll[:] = torch.cat(self.log_probs, dim=0)
            self.done_roll[:] = torch.cat(self.dones, dim=0)
        
        else: # done true
            diff = self.args.seq_len - self.size
            
            self.obs_roll[:] = torch.cat(self.obs + [self.next_obs] * (diff+1), dim=0)
            self.action_roll[:] = torch.cat(self.actions + [self.actions[-1]] * diff, dim=0)
            self.reward_roll[:] = torch.cat(self.rewards + [self.rewards[-1]] * diff, dim=0)
            self.log_prob_roll[:] = torch.cat(self.log_probs + [self.log_probs[-1]] * diff, dim=0)
            self.done_roll[:] = torch.cat(self.dones + [self.dones[-1]] * diff, dim=0)
            
        lstm_states = self.lstm_hidden_states[0] # need only initial
        self.hidden_state_roll[:] = lstm_states[0] # hidden
        self.cell_state_roll[:] = lstm_states[1]   # cell