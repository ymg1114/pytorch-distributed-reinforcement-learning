from contextlib import contextmanager

import torch.multiprocessing as mp


class Lock:
    def __init__(self):
        self._lock = mp.Lock()
    
    # 서로 다른 프로세스 간, 데이터 오염을 방지하기 위함
    @contextmanager
    def lock(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
        return
    
    def get(self, queue):
        with self.lock():
            data = queue.get()
        return data
    
    def put(self, queue, data):
        with self.lock():
            queue.put(data)
        return
