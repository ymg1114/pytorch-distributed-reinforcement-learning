import asyncio

from heapq import heappush, heappop
from buffers.trajectory import Trajectory


class RolloutAssembler:
    def __init__(self, args):
        self.args = args
        self.seq_len = args.seq_len
        self.roll_q = dict()
        # self.ready_roll = list()
        self.ready_roll = asyncio.Queue(1024)

    # async def pop(self):
    #     while len(self.ready_roll) > 0:

    #         await asyncio.sleep(0.01)
    #         return heappop(self.ready_roll)

    async def pop(self):
        while self.ready_roll.qsize() > 0:

            await asyncio.sleep(0.01)
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

        _id = data["id"]  # epi. 별 독립적인 게임 id

        if _id in self.roll_q:
            await self.roll_q[_id].put(data)
        else:
            if len(self.roll_q) > 0:
                __id, _ = heappop(
                    list(self.roll_q.items())
                )  # 데이터의 크기/값 이 가장 작은 Trajectory 추출
                # TODO: heappop이 의도대로 작동하는지 검토 필요
                t = self.roll_q.pop(__id)
                data["is_fir"] = True
            else:
                t = Trajectory(self.seq_len)  # Trajectory 초기화를 통한 할당
            await t.put(data)
            self.roll_q[_id] = t

        # 롤아웃 시퀀스 길이가 충족된 경우
        if self.roll_q[_id].len >= self.seq_len:
            # heappush(self.ready_roll, self.roll_q.pop(_id))
            await self.ready_roll.put(self.roll_q.pop(_id))
