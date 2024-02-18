import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import zmq
import gym
import torch
import numpy as np

from utils.utils import Protocol, encode, decode, obs_preprocess, W_IP


class Env:
    def __init__(self, args):
        self.args = args
        self._env = gym.make(args.env)

    def reset(self):
        obs, _ = self._env.reset()
        return obs_preprocess(obs, self.args.need_conv)

    def step(self, act):
        obs, rew, done, _, _ = self._env.step(act)
        return obs_preprocess(obs, self.args.need_conv), rew, done


class Worker:
    def __init__(self, args, model, worker_name, port, obs_shape):
        self.args = args
        self.device = args.device  # cpu
        self.env = Env(args)

        self.model = model
        self.worker_name = worker_name

        self.zeromq_set(port)

    def __del__(self): # 소멸자
        self.pub_socket.close()
        self.sub_socket.close()

    def zeromq_set(self, port):
        context = zmq.Context()

        # worker <-> manager
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{W_IP}:{port}")  # publish rollout, stat

        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(
            f"tcp://{W_IP}:{self.args.learner_port+1}"
        )  # subscribe model
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    def pub_rollout(self, **step_data):
        self.pub_socket.send_multipart([*encode(Protocol.Rollout, step_data)])
        # print(f"worker_name: {self.worker_name} pub rollout to manager!")

    # NO-BLOCK
    def req_model(self):
        try:
            protocol, data = decode(*self.sub_socket.recv_multipart(flags=zmq.NOBLOCK))
            if protocol is Protocol.Model:
                model_state_dict = {k: v.to("cpu") for k, v in data.items()}
                if model_state_dict:
                    self.model.load_state_dict(
                        model_state_dict
                    )  # reload learned-model from learner
                    # print( f'{self.worker_name}: Received fresh model from learner !' )
        except zmq.Again as e:
            # print("No model-weight received yet")
            pass

    def pub_stat(self):
        stat = {}
        stat.update({"epi_rew": self.epi_rew})
        self.pub_socket.send_multipart([*encode(Protocol.Stat, stat)])

    def collect_rolloutdata(self):
        print("Build Environment for {}".format(self.worker_name))

        self.num_epi = 0
        while True:
            obs = self.env.reset()
            _id = int(np.random.random() * 10000000)
            # print(f"worker_name: {self.worker_name}, obs: {obs}")
            lstm_hx = (
                torch.zeros(self.args.hidden_size),
                torch.zeros(self.args.hidden_size),
            )  # (h_s, c_s) / (hidden,)
            self.epi_rew = 0

            is_fir = True  # first frame
            for _ in range(self.args.time_horizon):
                self.req_model()  # every-step
                act, logits, lstm_hx_next = self.model.act(obs, lstm_hx)
                next_obs, rew, done = self.env.step(act.item())

                self.epi_rew += rew

                step_data = {
                    "obs": obs,  # (c, h, w) or (D,)
                    "act": act.view(-1),  # (1,) / not one-hot, but action index
                    "rew": torch.from_numpy(
                        np.array([rew * self.args.reward_scale])
                    ),  # (1,)
                    "logits": logits,
                    "is_fir": torch.FloatTensor([1.0 if is_fir else 0.0]),  # (1,),
                    "done": torch.FloatTensor([1.0 if done else 0.0]),  # (1,),
                    "hx": lstm_hx[0],  # (hidden,)
                    "cx": lstm_hx[1],  # (hidden,)
                    "id": _id,
                }

                self.pub_rollout(**step_data)

                is_fir = False
                obs = next_obs
                lstm_hx = lstm_hx_next

                # time.sleep(0.1)

                if done:
                    break

            self.pub_stat()
            # self.req_model() # every-epi
            self.epi_rew = 0
            self.num_epi += 1
