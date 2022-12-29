import os, sys
import cv2
import json
import torch
import time
import psutil
import pickle
import blosc2
import torchvision.transforms as T
import torch.multiprocessing as mp

from enum import Enum, auto
from pathlib import Path
from signal import SIGTERM # or SIGKILL
from sys import platform
from numpy import dtype
from types import SimpleNamespace


utils = os.path.join(os.getcwd(), "utils", 'parameters.json')
with open(utils) as f:
    _p = json.load(f)
    Params = SimpleNamespace(**_p)
    
    
# def str2bool(v):
#       return v.lower() in ("yes", "Yes", "true", "True, "T", t", "1")
    
def mul(*args):
    _val = 1.0
    for e in args:
        _val *= e
    return _val
    
        
def SaveErrorLog(error: str, log_dir: str):
    current_time = time.strftime("[%Y_%m_%d][%H_%M_%S]", time.localtime(time.time()))
    # log_dst = os.path.join(log_dir, f"error_log_{current_time}.txt")
    dir = Path(log_dir)
    error_log = dir / f"error_log_{current_time}.txt"
    error_log.write_text(f"{error}\n")
    return

    
def obs_preprocess(obs, need_conv):
    global Params
    if Params.gray:
        transform = T.Compose([
            T.Grayscale(num_out_channels=1),
            # T.Resize( (p.H, p.W) ),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                              )
    else:
        transform = T.Compose([
            # T.Resize((p.H, p.W)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                              )
    if need_conv:
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGRA2RGB)    
        obs = cv2.resize(obs, dsize=(Params.H, Params.W), interpolation=cv2.INTER_AREA)
        # obs = obs.transpose((2, 0, 1))
        return transform(obs).unsqueeze(0).to(torch.float32) # (H, W, C) -> (1, C, H, W)
    
    else:
        return torch.from_numpy(obs).unsqueeze(0).to(torch.float32) # (D) -> (1, D)


class ParameterServer():
    def __init__(self, lock):
        self.lock = lock
        self.weight = None

    def pull(self):
        with self.lock:
            return self.weight

    def push(self, weigth):
        with self.lock:
            self.weight = weigth
            
            
class Protocol(Enum):
    Model = auto()
    Rollout = auto()
    Stat = auto()
    
    
# def KillProcesses():    
#     # python main.py로 실행된 프로세스를 찾음 
#     for proc in psutil.process_iter(): 
#         try: # 프로세스 이름, PID값 가져오기 
#             processName = proc.name() 
#             processID   = proc.pid 
#             if processName[:6] == "python": # 윈도우는 python.exe로 올라옴 
#                 commandLine = proc.cmdline() 
                
#                 # 동일한 프로세스 확인. code 확인 
#                 if 'main.py' in commandLine: 
#                     parent_pid = processID # PID 
#                     parent = psutil.Process(parent_pid) # PID 찾기 
                    
#                     for child in parent.children(recursive=True): #자식-부모 종료 
#                         child.kill() 
                        
#                     parent.kill() 
#             else: 
#                 print(processName, ' ', proc.cmdline(), ' - ', processID) 
            
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): #예외처리 
#             pass
        
        
# def KillProcesses():
#     global Params
#     WORKER_PORTS = [Params.worker_port]
#     LEARNER_PORTS = [Params.learner_port, Params.learner_port+1]
    
#     for proc in psutil.process_iter():
#         for conns in proc.connections(kind='inet'):
#             for port in WORKER_PORTS+LEARNER_PORTS:
#                 if conns.laddr.port == port:
#                     try:
#                         proc.send_signal(SIGTERM) # or SIGKILL
#                     except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): #예외처리 
#                         pass


def KillProcesses(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()    
        
        
def encode(protocol, data):
    return pickle.dumps(protocol), blosc2.compress(pickle.dumps(data), clevel=1, cname='zstd')


def decode(protocol, data):
    return pickle.loads(protocol), pickle.loads(blosc2.decompress(data))


if __name__ == "__main__":
    KillProcesses()