import torch
import time
import zmq
import numpy as np
import torch.multiprocessing as mp

local = "127.0.0.1"

class rep_Manager():
    def __init__(self, WORKER_PORTS):
        self.zeromq_set( WORKER_PORTS )
 
    def zeromq_set(self, WORKER_PORTS):
        # manager <-> worker 
        context = zmq.Context()
        self.rep_socket = context.socket( zmq.REP ) # receive rollout-data
        for port in WORKER_PORTS:
            self.rep_socket.connect( f"tcp://{local}:{port}" )

    def rep_rollout_from_workers(self, q_workers):
        while True:
            rollout_data = self.rep_socket.recv_pyobj()
            q_workers.put( rollout_data )
            
            self.rep_socket.send_pyobj( "RECEIVED_ROLLOUT_DATA" )
            print( 'Received rollout-data from workers !' )