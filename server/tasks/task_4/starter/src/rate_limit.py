from collections import deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events = deque()

    def allow(self, timestamp: int) -> bool:
        while self._events and timestamp - self._events[0] >= self.window_seconds:
            self._events.popleft()
        if len(self._events) > self.limit:
            return False
        self._events.append(timestamp)
        return True
