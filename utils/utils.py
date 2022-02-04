import os, sys
import cv2
import json
import torch
import psutil
import pickle
import blosc2
import torchvision.transforms as T

from sys import platform
from numpy import dtype
from types import SimpleNamespace

# def str2bool(v):
#       return v.lower() in ("yes", "true", "t", "1")

utils = os.path.join(os.getcwd(), "utils", 'parameters.json')
with open( utils ) as f:
    p = json.load(f)
    p = SimpleNamespace(**p)
    
if p.gray:
    transform = T.Compose([
                        T.Grayscale(num_out_channels=1),
                        # T.Resize( (p.H, p.W) ),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                          )
else:
    transform = T.Compose([
                        # T.Resize( (p.H, p.W) ),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                          )
    
def obs_preprocess(obs, need_conv):
    if need_conv:
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGRA2RGB)    
        obs = cv2.resize(obs, dsize=(p.H, p.W), interpolation=cv2.INTER_AREA)
        # obs = obs.transpose( (2, 0, 1) )
        return transform( obs ).unsqueeze(0).to(torch.float32) # (H, W, C) -> (1, C, H, W)
    
    else:
        return torch.from_numpy( obs ).unsqueeze(0).to(torch.float32) # (D) -> (1, D)

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
            
# def kill_processes():    
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
        
        
def kill_processes():
    # WORKER_PORTS = [ p.worker_port ]
    # LEARNER_PORTS = [ p.learner_port, p.learner_port+1  ]

    # for port in WORKER_PORTS+LEARNER_PORTS:
    #     os.system(f'taskkill /f /pid {port}')

    parent = psutil.Process( os.getppid() )
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()
        
        
def encode(filter, data):
    return blosc2.compress2( pickle.dumps(filter) ), blosc2.compress2( pickle.dumps(data) )

def decode(filter, data):
    return pickle.loads( blosc2.decompress2(filter) ), pickle.loads( blosc2.decompress2(data) )