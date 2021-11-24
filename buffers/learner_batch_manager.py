import torch
import zmq
import numpy as np

local = "127.0.0.1"

class learner_batch_Manager():
    def __init__(self, args):
        self.args = args        
        self.zeromq_set()
        
    def zeromq_set(self):
        # learner <-> manager
        context = zmq.Context()
        self.sub_socket = context.socket( zmq.SUB ) # subscribe batch-data
        self.sub_socket.connect( f"tcp://{local}:{self.args.learner_port}" )
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    def sub_batch_from_manager(self, q_batchs):
        while True:
            batch_data = self.sub_socket.recv_pyobj()
            q_batchs.put( batch_data )  
            # print( 'Received batch-data from manager !' )