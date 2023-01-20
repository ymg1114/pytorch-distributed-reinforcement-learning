import torch
import zmq
import time
import asyncio
import numpy as np
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

from utils.utils import Protocol, mul, encode, decode
from utils.lock import Lock
from threading import Thread

local = "127.0.0.1"


L = Lock() 


class Manager():
    def __init__(self, args, worker_port, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        # self.device = self.args.device
        self.device = torch.device('cpu')
        
        self.data_q = deque(maxlen=1024)

        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set(worker_port)
        
    def zeromq_set(self, worker_port):
        context = zmq.asyncio.Context()
        
        # worker <-> manager 
        self.sub_socket = context.socket(zmq.SUB) # subscribe rollout-data, stat-data
        self.sub_socket.bind(f"tcp://{local}:{worker_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        # manager <-> learner-storage
        self.pub_socket = context.socket(zmq.PUB) # publish batch-data, stat-data
        self.pub_socket.connect(f"tcp://{local}:{self.args.learner_port}")

    @staticmethod
    def process_stat(stat_list):
        mean_stat = {}
        for stat_dict in stat_list:
            for k, v in stat_dict.items():
                if not k in mean_stat:
                    mean_stat[k] = [v]
                else:
                    mean_stat[k].append(v)
                    
        mean_stat = {k: np.mean(v) for k, v in mean_stat.items()}
        return mean_stat

    async def sub_data(self):
        while True:
            protocol, data = decode(*(await self.sub_socket.recv_multipart()))
            if len(self.data_q) == self.data_q.maxlen:
                self.data_q.popleft() # FIFO
            self.data_q.append(protocol, data)
            
            await asyncio.sleep(0.01)
    
    async def pub_data(self):
        while True:
            if len(self.data_q) > 0:
                protocol, data = self.data_q.pop()
                if protocol is Protocol.Rollout:
                    await self.pub_socket.send_multipart([*encode(Protocol.Rollout, data)])
                    
                elif protocol is Protocol.Stat:
                    self.stat_list.append(data)
                    if len(self.stat_list) >= self.stat_log_len:
                        mean_stat = Manager.process_stat(self.stat_list)
                        await self.pub_socket.send_multipart([*encode(Protocol.Stat, {"log_len": len(self.stat_list), "mean_stat": mean_stat})])
                        self.stat_list = []
                else:
                    assert False, f"Wrong protocol: {protocol}"
                    
            await asyncio.sleep(0.01)
        
    async def data_chain(self):
        tasks = [asyncio.create_task(self.sub_data()), asyncio.create_task(self.pub_data())]
        await asyncio.gather(*tasks) 