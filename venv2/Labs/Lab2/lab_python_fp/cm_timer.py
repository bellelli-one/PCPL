import time
from contextlib import contextmanager


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exc_time = time.time() - self.start
        print(f"time: {exc_time}")

@contextmanager
def timer():
    start = time.time()
    yield
    exc_time = time.time() - start
    print(f"time: {exc_time}")


# with timer() as t:
#     time.sleep(5.5)

# with Timer() as t:
#     time.sleep(5.5)
