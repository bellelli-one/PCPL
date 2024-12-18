import random

def gen_random(c, s, e):
    vals = [random.randint(s, e) for _ in range(c)]
    yield vals


print(str(list(gen_random(5,1,3)))[1:-1])
# def fib(n):
#     a = 0
#     b = 1
#     for _ in range(n):
#         yield a
#         a, b = b, a + b

# print(list(fib(10)))