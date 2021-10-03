import torch.nn as nn
import torch as torch
from torch.distributions import Categorical
from enum import Enum

# For torch.jit
from typing import Tuple

def _output_size_conv2d(conv, size):
    """
    Computes the output size of the convolution for an input size
    """
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
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()

        self.in_ch = in_ch

        self.conv1 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        out += residual
        return out


class BaseBlock(nn.Module):
    """
    Blocks for a residual model for reinforcement learning task as presented in He. and al, 2016
    """

    def __init__(self, in_ch, out_ch):
        super(BaseBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=self.out_ch, 
            kernel_size=3, 
            stride=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual = ResidualBlock(in_ch=self.out_ch)

        self._body = nn.Sequential(self.conv, self.max_pool, self.residual)

    def forward(self, x):
        x = self._body(x)
        return x

    def output_size(self, size):
        size = _output_size_conv2d(self.conv, size)
        size = _output_size_conv2d(self.max_pool, size)
        return size


class DeepConv(nn.Module):
    """
    Deeper model that uses 12 convolutions with residual blocks
    """

    def __init__(self, c):
        """c is the number of channels in the input tensor"""
        super(DeepConv, self).__init__(self)

        self.block_1 = BaseBlock(c, 16)
        self.block_2 = BaseBlock(16, 32)
        self.block_3 = BaseBlock(32, 32)

        self._body = nn.Sequential(self.block_1, self.block_2, self.block_3)

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = self.block_1.output_size(size)
        size = self.block_2.output_size(size)
        size = self.block_3.output_size(size)
        return size


class ShallowConv(nn.Module):
    """
    Shallow model that uses only 3 convolutions
    """

    def __init__(self, c):
        super(ShallowConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        self._body = nn.Sequential(
            self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU()
        )

    def forward(self, x):
        return self._body(x)

    def output_size(self, size):
        size = _output_size_conv2d(self.conv1, size)
        size = _output_size_conv2d(self.conv2, size)
        size = _output_size_conv2d(self.conv3, size)
        return size


class BodyType(Enum):
    """Enum to specify the body of out network"""

    SHALLOW = 1
    DEEP = 2


class ActorCriticLSTM(nn.Module):
    """Actor Critic network with an LSTM on top, and accelerated with jit"""

    __constants__ = ["flatten_dim", "hidden_size", "n_outputs", "sequence_length"]

    def __init__(self, h, w, c, n_outputs, sequence_length=1, body=BodyType.DEEP):
        """You can have several types of body as long as they implement the size function"""
        super(ActorCriticLSTM, self).__init__()

        # Keeping some infos
        self.n_outputs = n_outputs
        self.input_size = (c, h, w)
        self.hidden_size = 256
        self.sequence_length = sequence_length

        # Sequential baseblocks
        if body == BodyType.SHALLOW:
            self.convs = ShallowConv(c)
        elif body == BodyType.DEEP:
            self.convs = DeepConv(c)
        else:
            raise AttributeError("The body type is not valid")

        conv_out = self.convs.output_size((h, w))
        self.flatten_dim = int(32 * conv_out[0] * conv_out[1])

        # Fully connected layers
        self.flatten = Flatten()
        self.fc = nn.Linear(in_features=self.flatten_dim, out_features=self.hidden_size)

        self.body = nn.Sequential(self.convs, Flatten(), nn.ReLU(), self.fc, nn.ReLU())

        # LSTM (Memory Layer)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1
        )

        # Allocate tensors as contiguous on GPU memory
        self.lstm.flatten_parameters()

        # Fully conected layers for value and policy
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

        # Using those distributions doesn't affect the gradient calculus
        # see https://arxiv.org/abs/1506.05254 for more infos
        self.dist = Categorical

    @torch.jit.export
    def act(self, obs: torch.Tensor, lstm_hxs: Tuple[torch.Tensor, torch.Tensor]):
        """Performs an one-step prediction with detached gradients"""
        # x: (batch, feat)
        x = self.body.forward(obs) 

        # x: (1, batch, feat)
        x = x.unsqueeze(0)

        x, lstm_hxs = self.lstm(x, lstm_hxs)
        
        # x: (batch, feat)
        x = x.squeeze(0)

        logits = self.logits(x)

        action, log_prob = self._act_dist(logits)

        lstm_hxs[0].detach_()
        lstm_hxs[1].detach_()

        return action.detach(), log_prob.detach(), lstm_hxs

    @torch.jit.ignore
    def _act_dist(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    @torch.jit.export
    def act_greedy(
        self, obs: torch.Tensor, lstm_hxs: Tuple[torch.Tensor, torch.Tensor]
    ):
        """Performs an one-step prediction with detached gradients"""
        # x : (batch, input_size)
        x = self.body.forward(obs)

        x = x.unsqueeze(0)
        # x : (1, batch, input_size)

        x, lstm_hxs = self.lstm(x, lstm_hxs)
        x = x.squeeze(0)

        # logits : (1, batch)
        logits = self.logits(x)
        action = torch.argmax(logits, dim=1)

        lstm_hxs[0].detach_()
        lstm_hxs[1].detach_()

        return action.detach(), lstm_hxs

    def forward(
        self, obs, lstm_hxs: Tuple[torch.Tensor, torch.Tensor], mask, behaviour_actions
    ):
        """
        obs : (seq, batch, c, h, w)
        lstm_hxs : ( (1, batch, hidden), (1, batch, hidden) )
        mask : (seq, batch, 1)
        behaviour_actions : (seq, batch, 1) / not one-hot, but action index
        """
        # Check the dimentions
        seq, batch, c, h, w = obs.size() 

        # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)

        obs = obs.view(seq * batch, c, h, w)
        x = self.body.forward(obs)
        x = x.view(seq, batch, self.hidden_size) # (seq, batch, hidden_size)

        x, lstm_hxs = self.lstm(x, lstm_hxs)
        lstm_hxs[0].detach_()
        lstm_hxs[1].detach_()
        
        x = x.view(seq * batch, self.hidden_size)                 # (seq*batch, hidden_size)
        
        behaviour_actions = behaviour_actions.view(seq*batch, 1)  # Shape for dist
                                                                  # (seq*batch, num_actions) / one-hot
                                                                  # (seq*batch, 1) / action index
        target_value = self.value(x) # (seq*batch, 1)
        logits = self.logits(x)      # (seq*batch, num_actions)

        target_log_probs, target_entropy = self._forward_dist(logits, behaviour_actions)

        target_log_probs = target_log_probs.view(seq, batch, 1)
        target_entropy = target_entropy.view(seq, batch, 1)
        target_value = target_value.view(seq, batch, 1)

        return target_log_probs, target_entropy, target_value, lstm_hxs

    @torch.jit.ignore
    def _forward_dist(self, logits, behaviour_actions):
        dist = self.dist(logits=logits)
        target_log_probs = dist.log_prob(behaviour_actions)
        target_entropy = dist.entropy()
        return target_log_probs, target_entropy