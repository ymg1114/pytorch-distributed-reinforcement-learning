import asyncio


class Trajectory:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        # self.data = list()
        self.data = asyncio.Queue(seq_len)

    @property
    def len(self):
        # return len(self.data)
        return self.data.qsize()

    async def put(self, data):
        # self.data.append(data)
        await self.data.put(data)

    async def get(self):
        return await self.data.get()
