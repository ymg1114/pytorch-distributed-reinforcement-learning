import uuid
import time
import zmq
import asyncio
import torch
import torch.jit as jit
import numpy as np

from agents.worker_module.env_maker import EnvBase

from utils.utils import Protocol, encode, decode, W_IP


class Worker:
    def __init__(
        self, args, model, worker_name, stop_event, port, obs_shape, heartbeat=None
    ):
        self.args = args
        self.device = args.device  # cpu
        self.env = EnvBase(args)

        # dummy_obs = torch.zeros(*obs_shape, requires_grad=False)
        # dummy_lstm_hx = (
        #     torch.zeros(self.args.hidden_size, requires_grad=False),
        #     torch.zeros(self.args.hidden_size, requires_grad=False),
        # )
        # self.model = jit.script(model.actor.to(torch.device("cpu")))
        # standalone_act = lambda obs, lstm_hx: self.model.act(obs, lstm_hx)
        # self.JitAct = jit.trace(
        #     standalone_act, (dummy_obs.detach(), dummy_lstm_hx.detach())
        # )
        self.model = model.actor.to(torch.device("cpu")).eval()
        self.stop_event = stop_event
        self.worker_name = worker_name
        self.heartbeat = heartbeat

        self.zeromq_set(port)

    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, port):
        context = zmq.asyncio.Context()

        # worker <-> manager
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{W_IP}:{port}")  # publish rollout, stat

        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(
            f"tcp://{W_IP}:{self.args.learner_port+1}"
        )  # subscribe model
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def pub_rollout(self, **step_data):
        await self.pub_socket.send_multipart([*encode(Protocol.Rollout, step_data)])
        # print(f"worker_name: {self.worker_name} pub rollout to manager!")

    async def req_model(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Model:
                model_state_dict = {k: v.to("cpu") for k, v in data.items()}
                if model_state_dict:
                    self.model.load_state_dict(
                        model_state_dict
                    )  # reload learned-model from learner

            await asyncio.sleep(0.1)

    async def pub_stat(self):
        await self.pub_socket.send_multipart([*encode(Protocol.Stat, self.epi_rew)])
        print(
            f"worker_name: {self.worker_name} epi_rew: {self.epi_rew} pub stat to manager!"
        )

    async def life_cycle_chain(self):
        tasks = [
            asyncio.create_task(self.collect_rolloutdata()),
            asyncio.create_task(self.req_model()),
        ]
        await asyncio.gather(*tasks)

    async def collect_rolloutdata(self):
        print("Build Environment for {}".format(self.worker_name))

        self.num_epi = 0
        while not self.stop_event.is_set():
            obs = self.env.reset()
            # _id = uuid.uuid4().int
            _id = str(uuid.uuid4())  # 고유한 난수 생성

            # print(f"worker_name: {self.worker_name}, obs: {obs}")
            lstm_hx = (
                torch.zeros(self.args.hidden_size),
                torch.zeros(self.args.hidden_size),
            )  # (h_s, c_s) / (hidden,)
            self.epi_rew = 0

            is_fir = True  # first frame
            for _ in range(self.args.time_horizon):
                act, logits, log_prob, lstm_hx_next = self.model.act(obs, lstm_hx)
                next_obs, rew, done = self.env.step(act.item())

                self.epi_rew += rew

                step_data = {
                    "obs": obs,  # (c, h, w) or (D,)
                    "act": act.view(-1),  # (1,) / not one-hot, but action index
                    "rew": torch.from_numpy(
                        np.array([rew * self.args.reward_scale])
                    ),  # (1,)
                    "logits": logits,
                    "log_prob": log_prob.view(-1),  # (1,) / scalar
                    "is_fir": torch.FloatTensor([1.0 if is_fir else 0.0]),  # (1,),
                    "done": torch.FloatTensor([1.0 if done else 0.0]),  # (1,),
                    "hx": lstm_hx[0],  # (hidden,)
                    "cx": lstm_hx[1],  # (hidden,)
                    "id": _id,
                }

                await self.pub_rollout(**step_data)

                is_fir = False
                obs = next_obs
                lstm_hx = lstm_hx_next

                await asyncio.sleep(0.05)

                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()

                if done:
                    break

            await self.pub_stat()

            self.epi_rew = 0
            self.num_epi += 1
