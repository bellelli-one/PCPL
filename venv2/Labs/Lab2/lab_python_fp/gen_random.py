import random

def gen_random(c, s, e):
    vals = [random.randint(s, e) for _ in range(c)]
    yield vals
