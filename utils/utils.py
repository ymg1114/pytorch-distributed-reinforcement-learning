import os, sys
import cv2
import json
import torch
import psutil
import torchvision.transforms as T

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
            
def kill_processes():
    WORKER_PORTS = [ p.worker_port ]
    LEARNER_PORTS = [ p.learner_port, p.learner_port+1  ]
    
    for port in WORKER_PORTS+LEARNER_PORTS:
        os.system(f'taskkill /f /pid {port}')
        
    parent = psutil.Process( os.getppid() )
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()
    
    
if __name__ == "__main__":
    kill_processes()