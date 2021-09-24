import torch

class WorkerRolloutStorage():
    def __init__(self):
        self.device = torch.device('cpu')
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.masks = []
        self.lstm_hidden_states = []
        
    def insert(self, obs, action, reward, log_prob, mask, lstm_hidden_state):
        self.obs.append( obs.to(self.device) )               # (1, c, h, w)
        self.actions.append( action.to(self.device) )        # (1, 1) / not one-hot, but action index
        self.rewards.append( reward.to(self.device) )        # (1, 1)
        self.log_probs.append( log_prob.to(self.device) )    # (1, 1)
        self.masks.append( mask.to(self.device) )            # (1, 1)
        
        lstm_hidden_state = ( lstm_hidden_state[0].to(self.device), lstm_hidden_state[1].to(self.device) ) 
        self.lstm_hidden_states.append( lstm_hidden_state )  # ( (1, 1, d_h), (1, 1, d_c) )