import threading
from typing import Optional, Any, List, Tuple


class RingBuffer:
    def __init__(self, size: int = 1024, fps: int = 30):
        self.size = size
        self.head = 1
        self.counter = 1
        self.mutex = threading.Lock()
        self.buf = [None] * size
        self.fps = fps

        self.last_rd_cnt = {'default': 0}

    def reset(self) -> Optional[Exception]:
        with self.mutex:
            self.counter = 1
            self.head = 1
            self.last_rd_cnt = {'default': 0}
            self.buf = [None] * self.size
            return None

    def push(self, item: Any) -> Optional[Exception]:
        """
        将数据项推入缓冲区。
        """
        with self.mutex:
            self.buf[self.head] = item
            self.head = (self.head + 1) % self.size
            self.counter += 1
            return None

    def peek(self, index: int = -1) -> Tuple[Optional[Any], int, Optional[Exception]]:
        """
        查看指定索引处的数据项。
        """
        if index >= self.counter:
            return None, -1, Exception("index out of range")
        if index < 0 or index <= (self.counter - self.size):
            index = self.counter - 1
        place = index % self.size
        return self.buf[place], index, None

    def pull(self, client_id: str = 'default') -> Tuple[List[Any], Optional[Exception]]:
        """
        返回缓冲区中所有未读的数据项。
        """
        if client_id not in self.last_rd_cnt.keys():
            self.last_rd_cnt[client_id] = 0

        with self.mutex:
            if self.counter <= self.last_rd_cnt[client_id] + 1:
                return [], None
            if self.last_rd_cnt[client_id] < 0 or self.last_rd_cnt[client_id] <= (self.counter - self.size):
                self.last_rd_cnt[client_id] = self.counter - 1
            ret = []
            for idx in range(self.last_rd_cnt[client_id] + 1, self.counter):
                item, _, err = self.peek(idx)
                if err is not None:
                    return [], err
                ret.append(item)
            self.last_rd_cnt[client_id] = self.counter - 1
            return ret, None

    def get_counter(self) -> int:
        return self.counter
