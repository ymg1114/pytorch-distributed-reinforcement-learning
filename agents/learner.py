import os, sys
import zmq
import torch
import pickle
import time
import numpy as np
from multiprocessing import shared_memory
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import deque
from utils.utils import Protocol, encode, decode
from utils.lock import Lock
from threading import Thread
from torch.optim import RMSprop, Adam
# from buffers.batch_buffer import LearnerBatchStorage

local = "127.0.0.1"

L = Lock()

class Learner():
    def __init__(self, args, sam_lock, dst_conn, model):
        self.args = args   
        self.sam_lock = sam_lock
        
        self.shm_ref = dst_conn.recv() # 반드시 공유메모리의 (이름, 버퍼)를 수신해야 함.
        assert hasattr(self.shm_ref, "obs_batch")
        assert hasattr(self.shm_ref, "action_batch")
        assert hasattr(self.shm_ref, "reward_batch")
        assert hasattr(self.shm_ref, "log_prob_batch")
        assert hasattr(self.shm_ref, "done_batch")
        assert hasattr(self.shm_ref, "hidden_state_batch")
        assert hasattr(self.shm_ref, "cell_state_batch")
        assert hasattr(self.shm_ref, "batch_num")
        
        self.device = self.args.device     
        self.model = model.to(args.device)
        # self.model.share_memory() # make other processes can assess
        # self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        
        # self.q_workers = mp.Queue(maxsize=args.batch_size) # q for multi-worker-rollout 
        # self.batch_buffer = LearnerBatchStorage(args, obs_shape)
        
        def make_gpu_batch(*args):
            to_gpu = lambda tensor: tensor.to(self.device)
            return tuple(map(to_gpu, args))
        self.make_gpu_batch = make_gpu_batch
        
        self.stat_list = []
        self.stat_log_len = 20
        self.zeromq_set()
        self.mean_cal_interval = 30

    def zeromq_set(self):
        context = zmq.Context()
        
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{local}:{self.args.learner_port+1}") # publish fresh learner-model

    def monitor_sh_batch_num(self):
        nshape = self.shm_ref["batch_num"][0]
        ndtype = self.shm_ref["batch_num"][1]
        nshm = self.shm_ref["batch_num"][2]
        bshm = self.shm_ref["batch_num"][3]
        
        # shm_obj = shared_memory.SharedMemory(name=nshm)
        self.sh_batch_num = np.frombuffer(buffer=bshm, dtype=ndtype)
        # self.sh_batch_num = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # self.sh_batch_num = np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        if (self.sh_batch_num >= self.args.batch_size).item():
            return True
        else:
            return False
       
    def get_np_array_from_sh_memory(self, target):
        nshape = self.shm_ref[target][0]
        ndtype = self.shm_ref[target][1]
        nshm = self.shm_ref[target][2]
        bshm = self.shm_ref[target][3]
        
        # shm_obj = shared_memory.SharedMemory(name=nshm)
        dst = np.empty(nshape, dtype=ndtype)
        # src = np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        src = np.frombuffer(buffer=bshm, dtype=ndtype)
        
        # return np.frombuffer(buffer=shm_obj.buf, dtype=ndtype)
        # return np.ndarray(nshape, dtype=ndtype, buffer=bshm)
        np.copyto(dst, src) # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst
    
    def get_batch_from_sh_memory(self):
        sq = self.args.seq_len
        bn = self.sh_batch_num
        
        sha = self.obs_shape
        hs = self.args.hidden_size
        # torch.tensor(s_lst, dtype=torch.float)
        to_torch = lambda nparray: torch.from_numpy(nparray)
        
        # (seq, batch, feat)
        sh_obs_bat = self.get_np_array_from_sh_memory("obs_batch").reshape(((sq+1), bn, *sha))
        sh_act_bat = self.get_np_array_from_sh_memory("action_batch").reshape((sq, bn, 1))
        sh_rew_bat = self.get_np_array_from_sh_memory("reward_batch").reshape((sq, bn, 1))
        sh_log_pb_bat = self.get_np_array_from_sh_memory("log_prob_batch").reshape((sq, bn, 1))
        sh_done_bat = self.get_np_array_from_sh_memory("done_batch").reshape((sq, bn, 1))
        sh_hsta_bat = self.get_np_array_from_sh_memory("hidden_state_batch").reshape((1, bn, hs))
        sh_csta_bat = self.get_np_array_from_sh_memory("cell_state_batch").reshape((1, bn, hs))
        
        return to_torch(sh_obs_bat), to_torch(sh_act_bat), to_torch(sh_rew_bat), to_torch(sh_log_pb_bat), to_torch(sh_done_bat), to_torch(sh_hsta_bat), to_torch(sh_csta_bat)
    
    # def data_subscriber(self, q_batchs):
    #     self.l_t = Thread(target=self.receive_data, args=(q_batchs,), daemon=True)
    #     self.l_t.start()
                
    # def receive_data(self, q_batchs):
    #     while True:
    #         protocol, data = decode(*self.sub_socket.recv_multipart())
    #         if protocol is Protocol.Batch:
    #             if q_batchs.qsize() == q_batchs._maxsize:
    #                 L.get(q_batchs)
    #             L.put(q_batchs, data)

    #         elif protocol is Protocol.Stat:
    #             self.stat_list.append(data["mean_stat"])
    #             if len(self.stat_list) >= self.stat_log_len:
    #                 mean_stat = self.process_stat()
    #                 self.log_stat_tensorboard({"log_len": self.stat_log_len, "mean_stat": mean_stat})
    #                 self.stat_list = []
    #         else:
    #             assert False, f"Wrong protocol: {protocol}"
                
    #         time.sleep(0.01)

    def pub_model(self, model_state_dict):        
        self.pub_socket.send_multipart([*encode(Protocol.Model,  model_state_dict)]) 
    
    #TODO: 이거 동작 안 함.. 흠.. 어떻게 하지??
    def log_loss_tensorboard(self, loss, detached_losses):
        self.writer.add_scalar('total-loss', float(loss.item()), self.idx)
        self.writer.add_scalar('original-value-loss', detached_losses["value-loss"], self.idx)
        self.writer.add_scalar('original-policy-loss', detached_losses["policy-loss"], self.idx)
        self.writer.add_scalar('original-policy-entropy', detached_losses["policy-entropy"], self.idx)

    def reset_batch_num(self):
        self.sh_batch_num[:] = 0
        return

    # PPO
    def learning(self):
        self.idx = 0
        
        while True:
            batch_args = None
            #TODO: 무리하게 세마포어 lock을 걸어버리는 게 아닐까..?
            with self.sam_lock(): 
                if self.monitor_sh_batch_num():
                    batch_args = self.get_batch_from_sh_memory()
                    self.reset_batch_num() # 공유메모리 저장 인덱스 (batch_num) 초기화
                    
                # if q_batchs.qsize() > 0:
                #     # Basically, mini-batch-learning (seq, batch, feat)
                #     obs, actions, rewards, log_probs, dones, hidden_states, cell_states = self.make_gpu_batch(*L.get(q_batchs))
                
            if batch_args is not None:
                # Basically, mini-batch-learning (seq, batch, feat)
                obs, actions, rewards, log_probs, dones, hidden_states, cell_states = self.make_gpu_batch(*batch_args)
                # epoch-learning
                for _ in range(self.args.K_epoch):
                    lstm_states = (hidden_states, cell_states) 
                    
                    # on-line model forwarding
                    target_log_probs, target_entropy, target_value, lstm_states = self.model(
                        obs,         # (seq+1, batch, c, h, w)
                        lstm_states, # ((1, batch, hidden_size), (1, batch, hidden_size))
                        actions
                        )     # (seq, batch, 1)         
                    
                    # masking
                    mask = 1 - dones * torch.roll(dones, 1, 0)
                    mask_next = 1 - dones
                                    
                    value_current = target_value[:-1] * mask     # current
                    value_next    = target_value[1:] * mask_next # next
                    
                    target_log_probs = target_log_probs * mask # current
                    log_probs = log_probs * mask               # current
                    target_entropy = target_entropy * mask     # current
                    
                    # td-target (value-target)
                    # delta for ppo-gae
                    td_target = rewards + self.args.gamma * value_next
                    delta = td_target - value_current
                    delta = delta.cpu().detach().numpy()
                    
                    # ppo-gae (advantage)
                    advantages = []
                    advantage_t = np.zeros(delta.shape[1:]) # Terminal: (seq, batch, d) -> (batch, d)
                    mask_row = mask_next.detach().cpu().numpy()
                    for (delta_row, m_r) in zip(delta[::-1], mask_row[::-1]):
                        advantage_t = delta_row + self.args.gamma * self.args.lmbda * m_r * advantage_t # recursive
                        advantages.append(advantage_t)
                    advantages.reverse()
                    # advantage = torch.stack(advantages, dim=0).to(torch.float)
                    advantage = torch.tensor(advantages, dtype=torch.float).to(self.device)

                    ratio = torch.exp(target_log_probs - log_probs)  # a/b == log(exp(a)-exp(b))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantage
                    
                    loss_policy = -torch.min(surr1, surr2).mean()
                    loss_value = F.smooth_l1_loss(value_current, td_target.detach()).mean()  
                    policy_entropy = target_entropy.mean()
                    
                    # Summing all the losses together
                    loss = self.args.policy_loss_coef*loss_policy + self.args.value_loss_coef*loss_value - self.args.entropy_coef*policy_entropy

                    # These are only used for the statistics
                    detached_losses = {
                        "policy-loss": loss_policy.detach().cpu(),
                        "value-loss": loss_value.detach().cpu(),
                        "policy-entropy": policy_entropy.detach().cpu(),
                    }
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    print("loss {:.3f} original_value_loss {:.3f} original_policy_loss {:.3f} original_policy_entropy {:.5f}".format( loss.item(), detached_losses["value-loss"], detached_losses["policy-loss"], detached_losses["policy-entropy"] ))
                    self.optimizer.step()
                    
                self.pub_model(self.model.state_dict())
                
                if (self.idx % self.args.loss_log_interval == 0):
                    self.log_loss_tensorboard(loss, detached_losses)

                if (self.idx % self.args.model_save_interval == 0):
                    torch.save(self.model, os.path.join(self.args.model_dir, f"impala_{self.idx}.pt"))
                
                self.idx+= 1