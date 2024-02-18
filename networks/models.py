import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class MlpLSTM(nn.Module):
    def __init__(self, f, n_outputs, sequence_length, hidden_size):
        super(MlpLSTM, self).__init__()
        self.input_size = f
        self.n_outputs = n_outputs
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),            
        )
        self.after_torso()
        
        self.CT = Categorical
        
    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )
        
        # value
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        # policy
        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def act(self, obs, lstm_hxs):
        x = self.body.forward(obs)  # x: (feat,)
        hx, cx = self.lstmcell(x, lstm_hxs)
        
        logits = self.logits(hx) # policy
        dist = self.get_dist(logits)

        return dist.sample().detach(), dist.logits.detach(), (hx.detach(), cx.detach())

    def get_dist(self, logits):
        probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)
        return self.CT(probs)

    def forward(self, obs, lstm_hxs, behaviour_acts):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        value = self.value(output)  # (batch, seq, 1)
        logits = self.logits(output)  # (batch, seq, num_acts)

        log_probs, entropy = self._forward_dist(logits, behaviour_acts)  # current

        log_probs = log_probs.view(batch, seq, 1)
        entropy = entropy.view(batch, seq, 1)
        # value = value.view(batch, seq, 1)

        return log_probs, entropy, value

    def _forward_dist(self, logits, behaviour_acts):
        dist = self.get_dist(logits) # (batch, seq, num_acts)
        
        log_probs = dist.log_prob(behaviour_acts.squeeze(-1))  # (batch, seq)
        entropy = dist.entropy()  # (batch, seq)

        return log_probs, entropy