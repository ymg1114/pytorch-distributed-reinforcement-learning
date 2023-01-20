import torch
import zmq
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
    def __init__(self, args, sam_lock, src_conn, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        # self.device = self.args.device
        self.device = torch.device('cpu')
        
        self.stat_list = []
        self.stat_log_len = 20
        self.sam_lock = sam_lock
        
        self.src_conn = src_conn
        self.zeromq_set()
        self.init_shared_memory()
        self.writer = SummaryWriter(log_dir=self.args.result_dir) # tensorboard-log
        
    def __del__(self):
        def del_sh(shm):
            shm.close()
            shm.unlink()
        del_sh(self.sh_obs_batch)
        del_sh(self.sh_action_batch)
        del_sh(self.sh_reward_batch)
        del_sh(self.sh_log_prob_batch)
        del_sh(self.sh_done_batch)
        del_sh(self.sh_hidden_state_batch)
        del_sh(self.sh_cell_state_batch)
        del_sh(self.sh_batch_num)
        
    def zeromq_set(self):
        context = zmq.Context()
        
        # learner-storage <-> manager
        self.sub_socket = context.socket(zmq.SUB) # subscribe batch-data, stat-data
        self.sub_socket.bind(f"tcp://{local}:{self.args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
                
    def init_shared_memory(self):
        _shm_ref = {}
        
        # obs_batch = torch.zeros(self.args.seq_len+1, self.args.batch_size, *self.obs_shape)
        obs_batch = np.zeros((self.args.seq_len+1)*self.args.batch_size*mul(self.obs_shape), dtype=np.float64)
        _ref = LearnerStorage.set_shared_memory(self, obs_batch, "obs_batch")
        _shm_ref.update({"obs_batch": _ref})
        
        # self.action_batch = np.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        action_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # not one-hot, but action index (scalar)
        _ref = LearnerStorage.set_shared_memory(self, action_batch, "action_batch")
        _shm_ref.update({"action_batch": _ref})
        
        reward_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        _ref = LearnerStorage.set_shared_memory(self, reward_batch, "reward_batch")
        _shm_ref.update({"reward_batch": _ref})
        
        log_prob_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        _ref = LearnerStorage.set_shared_memory(self, log_prob_batch, "log_prob_batch")
        _shm_ref.update({"log_prob_batch": _ref})
        
        done_batch = np.zeros(self.args.seq_len*self.args.batch_size*1, dtype=np.float64) # scalar
        _ref = LearnerStorage.set_shared_memory(self, done_batch, "done_batch")
        _shm_ref.update({"done_batch": _ref})
        
        hidden_state_batch = np.zeros(1*self.args.batch_size*self.args.hidden_size, dtype=np.float64)
        _ref = LearnerStorage.set_shared_memory(self, hidden_state_batch, "hidden_state_batch")
        _shm_ref.update({"hidden_state_batch": _ref})
        
        cell_state_batch = np.zeros(1*self.args.batch_size*self.args.hidden_size, dtype=np.float64)
        _ref = LearnerStorage.set_shared_memory(self, cell_state_batch, "cell_state_batch")
        _shm_ref.update({"cell_state_batch": _ref})
        
        batch_num = np.zeros(1, dtype=np.float64)
        _ref = LearnerStorage.set_shared_memory(self, batch_num, "batch_num")
        _shm_ref.update({"batch_num": _ref})
        self.reset_batch_num() # 공유메모리 저장 인덱스 (batch_num) 초기화
        
        self.src_conn.send(_shm_ref) # 반드시 공유메모리의 (이름, 버퍼)를 송신해야 함.
        self.src_conn.close()

        # self.sh_batch_num = mp.Value('i', 0) #TODO: 이 mp.Value, mp.Array로 만든 공유메모리의 레퍼런스 (주소) 획득을 어떻게 하지..?
        return

    @staticmethod
    def set_shared_memory(self, np_array, name):
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        # shm = mp.Array('f', np_array.shape)
        setattr(self, f"sh_{name}", np.frombuffer(buffer=shm.buf, dtype=np_array.dtype, count=-1))
        # setattr(self, f"sh_{name}_ref", shm.name)
        return np_array.shape, np_array.dtype, shm.name, shm.buf # 공유메모리의 (이름, 버퍼)
    
    def reset_batch_num(self):
        self.sh_batch_num[:] = 0
        return
    
    def set_data_to_shared_memory(self):
        while True:
            protocol, data = decode(*self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                with self.sam_lock():
                    self.make_batch(data)
                
            elif protocol is Protocol.Stat:
                self.stat_list.append(data["mean_stat"])
                if len(self.stat_list) >= self.stat_log_len:
                    mean_stat = self.process_stat()
                    self.log_stat_tensorboard({"log_len": self.stat_log_len, "mean_stat": mean_stat})
                    self.stat_list = []
            else:
                assert False, f"Wrong protocol: {protocol}"
                
            time.sleep(0.01)
            
    def process_stat(self):
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
        len       = data['log_len']
        data_dict = data['mean_stat']
        
        for k, v in data_dict.items():
            tag = f'worker/{len}-game-mean-stat-of-{k}'
            # x = self.idx
            x = self.log_stat_tensorboard.calls * len # global game counts
            y = v
            self.writer.add_scalar(tag, y, x)
            print(f'tag: {tag}, y: {y}, x: {x}')
            
    def make_batch(self, rollout):
        sq = self.args.seq_len
        bn = self.sh_batch_num
        
        sha = mul(self.obs_shape)
        hs = self.args.hidden_size
        
        if (bn < self.args.batch_size).item():
            obs          = rollout[0]
            action       = rollout[1]
            reward       = rollout[2]
            log_prob     = rollout[3]
            done         = rollout[4]
            hidden_state = rollout[5]
            cell_state   = rollout[6]

            # 공유메모리에 학습 데이터 적재
            self.sh_obs_batch[(sq+1)*bn*sha: (sq+1)*(bn+1)*sha] = obs.reshape(-1).astype(np.float64)
            self.sh_action_batch[sq*bn: sq*(bn+1)] = action.reshape(-1).astype(np.float64)
            self.sh_reward_batch[sq*bn: sq*(bn+1)] = reward.reshape(-1).astype(np.float64) 
            self.sh_log_prob_batch[sq*bn: sq*(bn+1)] = log_prob.reshape(-1).astype(np.float64)
            self.sh_done_batch[sq*bn: sq*(bn+1)] = done.reshape(-1).astype(np.float64)
            self.sh_hidden_state_batch[1*bn*hs: 1*(bn+1)*hs] = hidden_state.reshape(-1).astype(np.float64)
            self.sh_cell_state_batch[1*bn*hs: 1*(bn+1)*hs] = cell_state.reshape(-1).astype(np.float64)
            
            self.sh_batch_num += 1
        
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