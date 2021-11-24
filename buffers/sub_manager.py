import torch
import zmq
import numpy as np

local = "127.0.0.1"

class sub_Manager():
    def __init__(self, WORKER_PORTS):
        self.zeromq_set( WORKER_PORTS )
 
    def zeromq_set(self, WORKER_PORTS):
        # manager <-> worker 
        context = zmq.Context()
        self.sub_socket = context.socket( zmq.SUB ) # subscribe rollout-data
        for port in WORKER_PORTS:
            self.sub_socket.connect( f"tcp://{local}:{port}" )
        self.sub_socket.setsockopt_string( zmq.SUBSCRIBE, '' )

    def sub_rollout_from_workers(self, q_workers):
        while True:
            rollout_data = self.sub_socket.recv_pyobj()
            q_workers.put( rollout_data )
            # print( 'Received rollout-data from workers !' )