import torch
import zmq
import zmq.asyncio
import asyncio
import time
import numpy as np
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

from utils.utils import Protocol, mul, encode, decode
from utils.lock import Lock
from threading import Thread
from tensorboardX import SummaryWriter

local = "127.0.0.1"


def counted(f):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


class LearnerStorage():
    def __init__(self, args, sam_lock, datafram_keyword, queue, obs_shape):
        self.args = args
        self.datafram_keyword = datafram_keyword
        self.obs_shape = obs_shape
        # self.device = self.args.device
        self.device = torch.device('cpu')
        
        self.stat_list = []
        self.stat_log_len = 20
        self.sam_lock = sam_lock
        self._init = False
        
        self.batch_queue = queue
        
        self.zeromq_set()
        self.init_shared_memory()
        self.writer = SummaryWriter(log_dir=args.result_dir) # tensorboard-log
        
    # def __del__(self):
    #     if hasattr(self, "_init") and self._init is True:
    #         def del_sh(shm):
    #             shm.close()
    #             shm.unlink()
    #         del_sh(self.sh_obs_batch)
    #         del_sh(self.sh_action_batch)
    #         del_sh(self.sh_reward_batch)
    #         del_sh(self.sh_log_prob_batch)
    #         del_sh(self.sh_done_batch)
    #         del_sh(self.sh_hidden_state_batch)
    #         del_sh(self.sh_cell_state_batch)
    #         del_sh(self.sh_batch_num)
        
    def zeromq_set(self):
        context = zmq.asyncio.Context()
        
        # learner-storage <-> manager
        self.sub_socket = context.socket(zmq.SUB) # subscribe batch-data, stat-data
        self.sub_socket.bind(f"tcp://{local}:{self.args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
                
    def init_shared_memory(self):
        self.shm_ref = {}
        
        # obs_batch = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
        obs_batch = np.zeros((self.args.seq_len+1)*self.args.batch_size*mul(self.obs_shape), dtype=np.float64)
        self.sh_obs_batch = LearnerStorage.set_shared_memory(self, obs_batch, "obs_batch")

        # self.action_batch = np.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        action_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # not one-hot, but action index (scalar)
        self.sh_action_batch = LearnerStorage.set_shared_memory(self, action_batch, "action_batch")

        reward_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        self.sh_reward_batch = LearnerStorage.set_shared_memory(self, reward_batch, "reward_batch")

        log_prob_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        self.sh_log_prob_batch = LearnerStorage.set_shared_memory(self, log_prob_batch, "log_prob_batch")

        done_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        self.sh_done_batch = LearnerStorage.set_shared_memory(self, done_batch, "done_batch")

        hidden_state_batch = np.zeros(1*self.args.batch_size*self.args.hidden_size, dtype=np.float64)
        self.sh_hidden_state_batch = LearnerStorage.set_shared_memory(self, hidden_state_batch, "hidden_state_batch")

        cell_state_batch = np.zeros(1*self.args.batch_size*self.args.hidden_size, dtype=np.float64)
        self.sh_cell_state_batch = LearnerStorage.set_shared_memory(self, cell_state_batch, "cell_state_batch")

        self.sh_batch_num = mp.Value('i', 0)
        self.reset_batch_num() # 공유메모리 저장 인덱스 (batch_num) 초기화

        self._init = True
        return

    @staticmethod
    def set_shared_memory(self, np_array, name):
        assert name in self.datafram_keyword
        
        # shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        shm_array = mp.Array('d', len(np_array))
        # setattr(self, f"sh_{name}", np.frombuffer(buffer=shm.buf, dtype=np_array.dtype, count=-1))
        # setattr(self, f"sh_{name}_ref", shm.name)
        # return np_array.shape, np_array.dtype, shm.name, shm.buf # 공유메모리의 (이름, 버퍼)
        # return np.frombuffer(buffer=shm.buf, dtype=np_array.dtype, count=-1)
        return np.frombuffer(buffer=shm_array.get_obj(), dtype=np_array.dtype, count=-1)
    
    def reset_batch_num(self):
        self.sh_batch_num.value = 0
        return
    
    async def shared_memory_chain(self):
        self.proxy_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.set_batch_from_shared_memory()), 
            asyncio.create_task(self.put_batch_to_proxy_q()),
            asyncio.create_task(self.put_batch_to_batch_q()),
            ]
        await asyncio.gather(*tasks) 
    
    async def put_batch_to_proxy_q(self):
        while True:
            # with self.sam_lock():
            if self.monitor_sh_batch_num():
                batch_args = self.get_batch_from_sh_memory()
                await self.proxy_queue.put(batch_args)
                self.reset_batch_num() # 공유메모리 저장 인덱스 (batch_num) 초기화

            await asyncio.sleep(0.01)

    async def put_batch_to_batch_q(self):
        while True:
            # with self.sam_lock():
            batch_args = await self.proxy_queue.get()
            self.batch_queue.put(batch_args)

            await asyncio.sleep(0.01)

    async def set_batch_from_shared_memory(self):
        while True:
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                # with self.sam_lock():
                await self.make_batch(data)

            elif protocol is Protocol.Stat:
                self.stat_list.append(data["mean_stat"])
                if len(self.stat_list) >= self.stat_log_len:
                    mean_stat = await self.process_stat()
                    self.log_stat_tensorboard({"log_len": self.stat_log_len, "mean_stat": mean_stat})
                    self.stat_list = []

            else:
                assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.01)
            
    async def process_stat(self):
        mean_stat = {}
        for stat_dict in self.stat_list:
            for k, v in stat_dict.items():
                if not k in mean_stat:
                    mean_stat[k] = [v]
                else:
                    mean_stat[k].append(v)
                    
        mean_stat = {k: np.mean(v) for k, v in mean_stat.items()}
        return mean_stat
            
    @counted
    def log_stat_tensorboard(self, data):
        len = data['log_len']
        data_dict = data['mean_stat']
        
        for k, v in data_dict.items():
            tag = f'worker/{len}-game-mean-stat-of-{k}'
            # x = self.idx
            x = self.log_stat_tensorboard.calls * len # global game counts
            y = v
            self.writer.add_scalar(tag, y, x)
            print(f'tag: {tag}, y: {y}, x: {x}')
            
    async def make_batch(self, rollout):
        sq = self.args.seq_len
        bn = self.sh_batch_num.value
        
        sha = mul(self.obs_shape)
        hs = self.args.hidden_size

        _flatten = lambda obj: obj.numpy().reshape(-1).astype(np.float64)
        if (bn < self.args.batch_size):
            obs = rollout[0]
            action = rollout[1]
            reward = rollout[2]
            log_prob = rollout[3]
            done = rollout[4]
            hidden_state = rollout[5]
            cell_state = rollout[6]
  
            # 공유메모리에 학습 데이터 적재
            self.sh_obs_batch[(sq+1)*bn*sha: (sq+1)*(bn+1)*sha] = _flatten(obs)
            self.sh_action_batch[sq*bn: sq*(bn+1)] = _flatten(action)
            self.sh_reward_batch[sq*bn: sq*(bn+1)] = _flatten(reward)
            self.sh_log_prob_batch[sq*bn: sq*(bn+1)] = _flatten(log_prob)
            self.sh_done_batch[sq*bn: sq*(bn+1)] = _flatten(done)
            self.sh_hidden_state_batch[1*bn*hs: 1*(bn+1)*hs] = _flatten(hidden_state)
            self.sh_cell_state_batch[1*bn*hs: 1*(bn+1)*hs] = _flatten(cell_state)
            
            self.sh_batch_num.value += 1

    def monitor_sh_batch_num(self):
        # assert "batch_num" in self.datafram_keyword
        
        # nshape = self.shm_ref["batch_num"][0]
        # ndtype = self.shm_ref["batch_num"][1]
        # nshm = self.shm_ref["batch_num"][2]
        # bshm = self.shm_ref["batch_num"][3]
        
        # shm_obj = shared_memory.SharedMemory(name=nshm)
        # self.sh_batch_num = np.frombuffer(buffer=bshm, dtype=ndtype)
        # self.sh_batch_num = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # self.sh_batch_num = np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        # if (self.sh_batch_num >= self.args.batch_size).item():
        if self.sh_batch_num.value >= self.args.batch_size:
            return True
        else:
            return False
    
    @staticmethod
    def copy_to_ndarray(src):
        # shm_obj = shared_memory.SharedMemory(name=nshm)
        dst = np.empty(src.shape, dtype=src.dtype)
        # src = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # src = np.frombuffer(buffer=bshm, dtype=ndtype)
        
        # return np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # return np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        np.copyto(dst, src) # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst
    
    def get_batch_from_sh_memory(self):
        sq = self.args.seq_len
        bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        
        # torch.tensor(s_lst, dtype=torch.float)
        _to_torch = lambda nparray: torch.from_numpy(nparray).type(torch.float32)

        # (seq, batch, feat)
        sh_obs_bat = LearnerStorage.copy_to_ndarray(self.sh_obs_batch).reshape(((sq+1), bn, *sha))
        sh_act_bat = LearnerStorage.copy_to_ndarray(self.sh_action_batch).reshape((sq, bn, 1))
        sh_rew_bat = LearnerStorage.copy_to_ndarray(self.sh_reward_batch).reshape((sq, bn, 1))
        sh_log_pb_bat = LearnerStorage.copy_to_ndarray(self.sh_log_prob_batch).reshape((sq, bn, 1))
        sh_done_bat = LearnerStorage.copy_to_ndarray(self.sh_done_batch).reshape((sq, bn, 1))
        sh_hsta_bat = LearnerStorage.copy_to_ndarray(self.sh_hidden_state_batch).reshape((1, bn, hs))
        sh_csta_bat = LearnerStorage.copy_to_ndarray(self.sh_cell_state_batch).reshape((1, bn, hs))
        
        return _to_torch(sh_obs_bat), _to_torch(sh_act_bat), _to_torch(sh_rew_bat), _to_torch(sh_log_pb_bat), _to_torch(sh_done_bat), _to_torch(sh_hsta_bat), _to_torch(sh_csta_bat)
    
    # def get_np_array_from_sh_memory(self, target):
    #     assert target in self.datafram_keyword
        
    #     nshape = self.shm_ref[target][0]
    #     ndtype = self.shm_ref[target][1]
    #     nshm = self.shm_ref[target][2]
    #     bshm = self.shm_ref[target][3]
        
    #     # shm_obj = shared_memory.SharedMemory(name=nshm)
    #     dst = np.empty(nshape, dtype=ndtype)
    #     # src = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
    #     src = np.frombuffer(buffer=bshm, dtype=ndtype)
        
    #     # return np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
    #     # return np.ndarray(nshape, dtype=ndtype, buffer=bshm)
    #     np.copyto(dst, src) # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
    #     return dst
    
    # def get_batch_from_sh_memory(self):
    #     sq = self.args.seq_len
    #     bn = self.sh_batch_num
        
    #     sha = self.obs_shape
    #     hs = self.args.hidden_size
    #     # torch.tensor(s_lst, dtype=torch.float)
    #     to_torch = lambda nparray: torch.from_numpy(nparray)
        
    #     # (seq, batch, feat)
    #     sh_obs_bat = self.get_np_array_from_sh_memory("obs_batch").reshape(((sq+1), bn, *sha))
    #     sh_act_bat = self.get_np_array_from_sh_memory("action_batch").reshape((sq, bn, 1))
    #     sh_rew_bat = self.get_np_array_from_sh_memory("reward_batch").reshape((sq, bn, 1))
    #     sh_log_pb_bat = self.get_np_array_from_sh_memory("log_prob_batch").reshape((sq, bn, 1))
    #     sh_done_bat = self.get_np_array_from_sh_memory("done_batch").reshape((sq, bn, 1))
    #     sh_hsta_bat = self.get_np_array_from_sh_memory("hidden_state_batch").reshape((1, bn, hs))
    #     sh_csta_bat = self.get_np_array_from_sh_memory("cell_state_batch").reshape((1, bn, hs))
        
    #     return to_torch(sh_obs_bat), to_torch(sh_act_bat), to_torch(sh_rew_bat), to_torch(sh_log_pb_bat), to_torch(sh_done_bat), to_torch(sh_hsta_bat), to_torch(sh_csta_bat)
    
    # def reset_batch(self):
    #     self.obs_batch          = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
    #     # self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
    #     self.action_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # not one-hot, but action index (scalar)
    #     self.reward_batch       = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    #     self.log_prob_batch     = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    #     self.done_batch         = torch.zeros(self.args.seq_len, self.args.batch_size, 1) # scalar
    
    #     self.hidden_state_batch = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
    #     self.cell_state_batch   = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        
    #     self.batch_num = 0
            
    # def make_batch(self, q_workers):
    #     while True:
    #         if self.check_q(q_workers):
    #             for _ in range(self.args.batch_size):
    #                 rollout = L.get(q_workers)

    #                 obs          = rollout[0]
    #                 action       = rollout[1]
    #                 reward       = rollout[2]
    #                 log_prob     = rollout[3]
    #                 done         = rollout[4]
    #                 hidden_state = rollout[5]
    #                 cell_state   = rollout[6]
                    
    #                 self.obs_batch[:, self.batch_num] = obs
    #                 self.action_batch[:, self.batch_num] = action
    #                 self.reward_batch[:, self.batch_num] = reward 
    #                 self.log_prob_batch[:, self.batch_num] = log_prob
    #                 self.done_batch[:, self.batch_num] = done
    #                 self.hidden_state_batch[:, self.batch_num] = hidden_state
    #                 self.cell_state_batch[:, self.batch_num] = cell_state
                    
    #                 self.batch_num += 1
                    
    #             self.produce_batch()
    #             self.reset_batch()
                
    #         time.sleep(0.01)
            
    #     # if hasattr(self, "m_t") and self.m_t is not None:
    #     #     self.m_t.join()
                    
    # def produce_batch(self):
    #     o, a, r, log_p, done, h_s, c_s = self.get_batch()

    #     batch = (
    #         o.to(self.device), 
    #         a.to(self.device), 
    #         r.to(self.device), 
    #         log_p.to(self.device), 
    #         done.to(self.device), 
    #         h_s.to(self.device), 
    #         c_s.to(self.device)
    #         )
    #     self.pub_batch_to_learner(batch)
        
    # def get_batch(self):
    #     o = self.obs_batch[:, :self.args.batch_size]    # (seq, batch, feat)
    #     a = self.action_batch[:, :self.args.batch_size]
    #     r = self.reward_batch[:, :self.args.batch_size]
    #     log_p = self.log_prob_batch[:, :self.args.batch_size]
    #     done = self.done_batch[:, :self.args.batch_size]
        
    #     h_s = self.hidden_state_batch[:, :self.args.batch_size]
    #     c_s = self.cell_state_batch[:, :self.args.batch_size]
        
    #     return o, a, r, log_p, done, h_s, c_s