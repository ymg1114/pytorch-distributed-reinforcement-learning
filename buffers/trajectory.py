import asyncio


class Trajectory:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.data = asyncio.Queue(seq_len)

    @property
    def len(self):
        return self.data.qsize()

    async def put(self, data):
        await self.data.put(data)

    async def get(self):
        return await self.data.get()


class Trajectory2:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.data = list()

    @property
    def len(self):
        return len(self.data)

    def put(self, data):
        self.data.append(data)
        return

    def get(self):
        return self.data.pop(0)
