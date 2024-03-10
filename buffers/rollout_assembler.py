import torch
import time

from collections import defaultdict
from heapq import heappush, heappop
from buffers.trajectory import Trajectory, Trajectory2


def make_as_array(rollout_obj):
    assert rollout_obj.len > 0

    refrased_rollout_data = defaultdict(list)

    for rollout in rollout_obj.data:
        for key, value in rollout.items():
            if key != "id":  # 학습 데이터만 취급
                refrased_rollout_data[key].append(value)

    refrased_rollout_data = {
        k: torch.stack(v, 0) for k, v in refrased_rollout_data.items()
    }
    return refrased_rollout_data


class RolloutAssembler:
    def __init__(self, args, ready_roll):
        self.args = args
        self.seq_len = args.seq_len
        self.roll_q = dict()
        self.roll_q_done = dict()
        self.ready_roll = ready_roll

    async def pop(self):
        return await self.ready_roll.get()

    async def push(self, data):
        assert "obs" in data
        assert "act" in data
        assert "rew" in data
        assert "logits" in data
        assert "is_fir" in data
        assert "hx" in data
        assert "cx" in data
        assert "id" in data
        assert "done" in data

        _id = data["id"]  # epi. 별 독립적인 경기 id
        _done = data["done"]  # 경기 종료 여부.

        # Trajectory 객체 최초 생성 이후 0.5초가 지나면 삭제. policy-lag를 줄이기 위함.
        self.roll_q = {
            id: tj
            for id, tj in self.roll_q.items()
            if time.monotonic() - tj.gen_time < 0.5
        }

        if _id in self.roll_q:  # 기존 경기에 데이터 추가
            self.roll_q[_id].put(data)
        else:
            if len(self.roll_q_done) > 0:
                _, id_ = heappop(
                    [(tj.len, id) for id, tj in self.roll_q_done.items()]
                )  # 데이터의 크기 (roll 개수)가 가장 작은 Trajectory 추출
                tj_ = self.roll_q_done.pop(id_)
                data["is_fir"] = torch.FloatTensor([1.0])
            else:
                tj_ = Trajectory2(
                    self.seq_len, time.monotonic()
                )  # Trajectory 객체 신규 생성을 통한 할당

            tj_.put(data)
            self.roll_q[_id] = tj_

        # 롤아웃 시퀀스 길이가 충족된 경우
        if self.roll_q[_id].len >= self.seq_len:
            # heappush(self.ready_roll, self.roll_q.pop(_id))
            await self.ready_roll.put(make_as_array(self.roll_q.pop(_id)))
        else:
            # 롤아웃 시퀀스가 종료된 경우
            if _done:
                self.roll_q_done[_id] = self.roll_q.pop(_id)
