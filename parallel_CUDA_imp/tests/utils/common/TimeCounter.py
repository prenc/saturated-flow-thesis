import time


class TimeCounter:
    def __init__(self):
        self.elapsed_time = None

    def start(self):
        self.elapsed_time = int(time.time())

    def stop(self):
        self.elapsed_time = int(time.time() - self.elapsed_time)
