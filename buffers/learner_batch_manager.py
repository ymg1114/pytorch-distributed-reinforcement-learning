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

class learner_batch_Manager():
    def __init__(self, args):
        self.args = args        
        self.zeromq_set()
        
    def zeromq_set(self):
        # learner <-> manager
        context = zmq.Context()
        self.rep_socket = context.socket( zmq.REP )
        self.rep_socket.connect( f"tcp://{local}:{self.args.learner_port}" ) # receive batch-data

    def rep_batch_from_manager(self, q_batchs):
        while True:
            batch_data = self.rep_socket.recv_pyobj()
            q_batchs.put( batch_data )
            
            self.rep_socket.send_pyobj( "RECEIVED_BATCH_DATA" )
            print( 'Received batch-data from manager !' )
