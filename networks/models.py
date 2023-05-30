import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical
from enum import Enum

# For torch.jit
from typing import Tuple


def _output_size_conv2d(conv, size):
    o_size = np.array(size) + 2 * np.array(conv.padding)
    o_size -= np.array(conv.dilation) * (np.array(conv.kernel_size) - 1)
    o_size -= 1
    o_size = o_size / np.array(conv.stride) + 1
    return np.floor(o_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.module = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                self.out_ch, self.out_ch, kernel_size=3, stride=stride, padding=1
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or self.in_ch != self.out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=stride)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.module(x) + self.shortcut(x)
        x = self.relu(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self.module[0], size)
        size = _output_size_conv2d(self.module[2], size)
        return size

    def output_ch(self):
        return self.out_ch


class BaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BaseBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self._body = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            ResidualBlock(self.out_ch, self.out_ch),
        )

    def forward(self, x):
        x = self._body(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self._body[0], size)
        size = _output_size_conv2d(self._body[1], size)

        size = self._body[3].output_size(size)
        return size

    def output_ch(self):
        return self.out_ch


class DeepConv(nn.Module):
    def __init__(self, c):
        super(DeepConv, self).__init__(self)

        self._body = nn.Sequential(
            BaseBlock(c, 16), BaseBlock(16, 32), BaseBlock(32, 32)
        )

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = self.block_1.output_size(size)
        size = self.block_2.output_size(size)
        size = self.block_3.output_size(size)
        return size

    def output_ch(self):
        return 32


class ShallowConv(nn.Module):
    def __init__(self, c):
        super(ShallowConv, self).__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = _output_size_conv2d(self.conv1, size)
        size = _output_size_conv2d(self.conv2, size)
        size = _output_size_conv2d(self.conv3, size)
        return size

    def output_ch(self):
        return 32


class BodyType(Enum):
    SHALLOW = 1
    DEEP = 2


class ConvLSTM(nn.Module):
    def __init__(
        self, h, w, c, n_outputs, sequence_length, hidden_size, body=BodyType.SHALLOW
    ):
        super(ConvLSTM, self).__init__()
        self.input_size = (c, h, w)
        self.init_param(n_outputs, sequence_length, hidden_size)

        # Sequential baseblocks
        if body == BodyType.SHALLOW:
            self.convs = ShallowConv(c)
        elif body == BodyType.DEEP:
            self.convs = DeepConv(c)
        else:
            raise AttributeError("The body type is not valid")

        conv_out = self.convs.output_size((h, w))
        self.flatten_dim = int(self.convs.output_ch * conv_out[0] * conv_out[1])

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim, out_features=self.hidden_size)

        self.body = nn.Sequential(self.convs, Flatten(), nn.ReLU(), self.fc, nn.ReLU())
        self.after_torso()

    def init_param(self, n_outputs, sequence_length, hidden_size):
        # Keeping some infos
        self.n_outputs = n_outputs
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def after_torso(self):
        # LSTM (Memory Layer)
        # self.lstm = nn.LSTM(
        #     input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1
        # )
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # Allocate tensors as contiguous on GPU memory
        # self.lstm.flatten_parameters()

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

        # Using those distributions doesn't affect the gradient calculus
        self.CT = Categorical

    @torch.jit.export
    def act(self, obs, lstm_hxs):
        """Performs an one-step prediction with detached gradients"""

        x = self.body.forward(obs)  # x: (feat,)

        hx, cx = self.lstmcell(x, lstm_hxs)

        logits = self.logits(hx)

        act, logits = self._act_dist(logits)

        return act.detach(), logits.detach(), (hx.detach(), cx.detach())

    @torch.jit.ignore
    def _act_dist(self, logits):
        probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)
        dist = self.CT(probs)

        act = dist.sample()
        # log_prob = dist.log_prob(act)
        logits = dist.logits
        # self.CT(F.softmax(logits, dim=-1)).log_prob(act)

        return act, logits

    # 사용하지 않는 코드
    # @torch.jit.export
    # def act_greedy(self, obs, lstm_hxs):
    #     """Performs an one-step prediction with detached gradients"""
    #     # x: (batch, feat)
    #     x = self.body.forward(obs)

    #     # x: (1, batch, feat)
    #     x = x.unsqueeze(0)

    #     x, lstm_hxs = self.lstm(x, lstm_hxs)

    #     # x: (batch, feat)
    #     x = x.squeeze(0)

    #     # logits: (batch, num_acts)
    #     # act: (batch,)
    #     logits = self.logits(x)
    #     act = torch.argmax(logits, dim=1)

    #     lstm_hxs[0].detach_()
    #     lstm_hxs[1].detach_()

    #     return act.detach(), lstm_hxs

    # 사용하지 않는 코드
    # def forward(self, obs, lstm_hxs, behaviour_acts):
    #     """
    #     obs : (seq+1, batch, d)
    #     lstm_hxs : ((1, batch, hidden), (1, batch, hidden))
    #     behaviour_acts : (seq, batch, 1) / not one-hot, but action index
    #     """
    #     # Check the dimentions
    #     seq, batch, *d = obs.size()
    #     seq -= 1

    #     obs = obs.contiguous().view((seq + 1) * batch, *d)
    #     x = self.body.forward(obs)
    #     x = x.view((seq + 1), batch, self.hidden_size)  # (seq+1, batch, hidden_size)

    #     x, lstm_hxs = self.lstm(x, lstm_hxs)
    #     lstm_hxs[0].detach_()
    #     lstm_hxs[1].detach_()

    #     value = self.value(x)  # ((seq+1), batch, 1)
    #     logits = self.logits(x)  # ((seq+1), batch, num_acts)
    #     logits = logits[:-1]  # (seq, batch, num_acts)

    #     log_probs, entropy = self._forward_dist(
    #         logits, behaviour_acts
    #     )  # current

    #     log_probs = log_probs.view(seq, batch, 1)
    #     entropy = entropy.view(seq, batch, 1)
    #     value = value.view((seq + 1), batch, 1)

    #     return log_probs, entropy, value, lstm_hxs

    def forward(self, obs, lstm_hxs, behaviour_acts):
        """
        obs : (batch, seq, d)
        lstm_hxs : ((batch, hidden), (batch, hidden))
        behaviour_acts : (batch, seq, 1) / not one-hot, but action index
        """

        # Check the dimentions
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

    @torch.jit.ignore
    def _forward_dist(self, logits, behaviour_acts):
        probs = F.softmax(logits, dim=-1)  # (batch, seq, num_acts)
        dist = self.CT(probs)  # (batch, seq, num_acts)

        log_probs = dist.log_prob(behaviour_acts.squeeze(-1))  # (batch, seq)
        entropy = dist.entropy()  # (batch, seq)

        return log_probs, entropy


class MlpLSTM(ConvLSTM):
    def __init__(self, f, n_outputs, sequence_length, hidden_size):
        nn.Module.__init__(self)

        self.input_size = f
        self.init_param(n_outputs, sequence_length, hidden_size)

        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
        )
        self.after_torso()
