import torch
import time

from utils.lock import Lock

L = Lock()


class LearnerBatchStorage:
    def __init__(self, args, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.reset_batch()

    def reset_batch(self):
        self.obs = torch.zeros(
            self.args.seq_len + 1, self.args.batch_size, *self.obs_shape
        )
        # self.act       = torch.zeros(self.args.seq_len, self.args.batch_size, self.n_outputs) # one-hot
        self.act = torch.zeros(
            self.args.seq_len, self.args.batch_size, 1
        )  # not one-hot, but action index (scalar)
        self.rew = torch.zeros(self.args.seq_len, self.args.batch_size, 1)  # scalar
        self.logits = torch.zeros(self.args.seq_len, self.args.batch_size, 1)  # scalar
        self.is_fir = torch.zeros(self.args.seq_len, self.args.batch_size, 1)  # scalar

        self.hx = torch.zeros(1, self.args.batch_size, self.args.hidden_size)
        self.cx = torch.zeros(1, self.args.batch_size, self.args.hidden_size)

        self.batch_num = 0

    def produce_batch(self):
        o, a, r, logits, is_fir, h_s, c_s = self.get_batch()

        batch = (
            o.to(self.args.device),
            a.to(self.args.device),
            r.to(self.args.device),
            logits.to(self.args.device),
            is_fir.to(self.args.device),
            h_s.to(self.args.device),
            c_s.to(self.args.device),
        )

        return batch

    def get_batch(self):
        o = self.obs  # (seq, batch, feat)
        a = self.act
        r = self.rew
        logits = self.logits
        is_fir = self.is_fir

        h_s = self.hx
        c_s = self.cx

        return o, a, r, logits, is_fir, h_s, c_s

    def check_q(self, q_workers):
        if q_workers.qsize() == self.args.batch_size:
            return True
        else:
            return False

    def roll_to_batch(self, q_workers):
        for _ in range(self.args.batch_size):
            rollout = L.get(q_workers)

            obs = rollout[0]
            act = rollout[1]
            rew = rollout[2]
            logits = rollout[3]
            is_fir = rollout[4]
            hx = rollout[5]
            cx = rollout[6]

            self.obs[:, self.batch_num] = obs
            self.act[:, self.batch_num] = act
            self.rew[:, self.batch_num] = rew
            self.logits[:, self.batch_num] = logits
            self.is_fir[:, self.batch_num] = is_fir
            self.hx[:, self.batch_num] = hx
            self.cx[:, self.batch_num] = cx

            self.batch_num += 1

        batch = self.produce_batch()
        self.reset_batch()

        return batch
