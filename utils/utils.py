import os
import cv2
import json
import torch
import time
import platform
import psutil
import pickle
import blosc2
import numpy as np
import torchvision.transforms as T

from enum import Enum, auto
from pathlib import Path

from datetime import datetime
from signal import SIGTERM  # or SIGKILL
from types import SimpleNamespace
from tensorboardX import SummaryWriter

utils = os.path.join(os.getcwd(), "utils", "parameters.json")
with open(utils) as f:
    _p = json.load(f)
    Params = SimpleNamespace(**_p)


class MetaclassSingleton(type):
    _instance = {}
    def __call__(cls,*args,**kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(MetaclassSingleton, cls).__call__(*args,*kwargs)
        return cls._instance[cls]


class WriterClass(metaclass=MetaclassSingleton):
    dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
    result_dir = os.path.join("results", str(dt_string))
    model_dir = os.path.join(result_dir, "models")

    wr = SummaryWriter(log_dir=result_dir)  # tensorboard-log


LS_IP = "127.0.0.1" # 동일 서브넷 다른 머신 사용 가능.
L_IP = "127.0.0.1" # 동일 서브넷 다른 머신 사용 가능.
W_IP = "127.0.0.1" # 동일 서브넷 다른 머신 사용 가능.
M_IP = "127.0.0.1" # 동일 서브넷 다른 머신 사용 가능.


flatten = lambda obj: obj.numpy().reshape(-1).astype(np.float64)


to_torch = lambda nparray: torch.from_numpy(nparray).type(torch.float32)


def make_gpu_batch(*args, device):
    to_gpu = lambda tensor: tensor.to(device)
    return tuple(map(to_gpu, args))


def get_current_process_id():
    return os.getpid()


def get_process_id(name):
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == name:
            return proc.info["pid"]


def kill_process(pid):
    os.kill(pid, SIGTERM)
    
    
class KillSubProcesses:
    def __init__(self, processes):
        self.target_processes = processes
        self.os_system = platform.system()

    def __call__(self):
        assert hasattr(self, "target_processes")
        for p in self.target_processes:
            if self.os_system == "Windows":
                print("This is a Windows operating system.")
                p.terminate()  # Windows에서는 "관리자 권한" 이 필요.

            elif self.os_system == "Linux":
                print("This is a Linux operating system.")
                os.kill(p.pid, SIGTERM)    
    

def mul(shape_dim):
    _val = 1
    for e in shape_dim:
        _val *= e
    return _val


def counted(f):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


def SaveErrorLog(error: str, log_dir: str):
    current_time = time.strftime("[%Y_%m_%d][%H_%M_%S]", time.localtime(time.time()))
    # log_dst = os.path.join(log_dir, f"error_log_{current_time}.txt")
    dir = Path(log_dir)
    error_log = dir / f"error_log_{current_time}.txt"
    error_log.write_text(f"{error}\n")
    return


def obs_preprocess(obs, need_conv):
    global Params
    if need_conv:
        if Params.gray:
            transform = T.Compose(
                [
                    T.Grayscale(num_out_channels=1),
                    # T.Resize( (p.H, p.W) ),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            transform = T.Compose(
                [
                    # T.Resize((p.H, p.W)),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGRA2RGB)
        obs = cv2.resize(obs, dsize=(Params.H, Params.W), interpolation=cv2.INTER_AREA)
        # obs = obs.transpose((2, 0, 1))
        return transform(obs).to(torch.float32)  # (H, W, C) -> (C, H, W)

    else:
        return torch.from_numpy(obs).to(torch.float32)  # (D)


class Protocol(Enum):
    Model = auto()
    Rollout = auto()
    Stat = auto()


def KillProcesses(pid):
    parent = psutil.Process(pid)
    for child in parent.children(
        recursive=True
    ):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()


def encode(protocol, data):
    return pickle.dumps(protocol), blosc2.compress(pickle.dumps(data), clevel=1)


def decode(protocol, data):
    return pickle.loads(protocol), pickle.loads(blosc2.decompress(data))


if __name__ == "__main__":
    KillProcesses()
