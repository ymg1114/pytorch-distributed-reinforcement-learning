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

local = "127.0.0.1"

class learner_update_Manager():
    def __init__(self, args):
        self.args = args        
        self.zeromq_set()
        self.model_weight = None
        
    def zeromq_set(self):
        context = zmq.Context()
        self.rep_socket1 = context.socket( zmq.REP )
        self.rep_socket1.bind( f"tcp://{local}:{self.args.learner_port + 1}" ) # send fresh learner model

        context = zmq.Context()
        self.rep_socket2 = context.socket( zmq.REP )
        self.rep_socket2.bind( f"tcp://{local}:{self.args.learner_port + 2}" ) # send fresh learner model

    def rep_model_to_workers(self, q_model, model_state_dict):
        while True:
            _ = self.rep_socket1.recv_pyobj()
            
            if self.model_weight:
                self.rep_socket1.send_pyobj( model_state_dict )
            else:
                self.rep_socket1.send_pyobj( '' )
            print( 'Send fresh model to workers !' )
