import numpy as np
from numerical_diff import numerical_diff

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def function_2(x):
    return x[0]**2 + x[1]**2