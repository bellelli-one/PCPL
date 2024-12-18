import json
import sys
from print_result import print_result
from cm_timer import Timer
from gen_random import gen_random

path = "/home/a1/PCPL/venv2/Labs/Lab2/lab_python_fp/data_light.json"

# Необходимо в переменную path сохранить путь к файлу, который был передан при запуске сценария

with open(path) as f:
    data = json.load(f)

# Далее необходимо реализовать все функции по заданию, заменив `raise NotImplemented`
# Предполагается, что функции f1, f2, f3 будут реализованы в одну строку
# В реализации функции f4 может быть до 3 строк

@print_result
def f1(arg):
    return sorted(set(i["job-name"].lower() for i in arg))


@print_result
def f2(arg):
    return list(filter(lambda x: x.startswith("программист"), arg))


@print_result
def f3(arg):
    return list(map(lambda x: x + " с опытом Python", arg))

@print_result
def f4(arg):
    return ['{} зарплата {}'.format(job, salary) for job, salary in zip(arg, *gen_random(len(arg), 100000, 2000000))]


if __name__ == '__main__':
    with Timer():
        f4(f3(f2(f1(data))))