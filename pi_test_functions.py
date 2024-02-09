import numpy as np

def linear_sum(x):
    return x[:, 0] + x[:, 1]


def exponential_difference(x):
    return 2 * np.exp(-x[:, 0] - x[:, 1])


def exponential_interaction(x):
    return 2 * np.exp(-x[:, 0] - x[:, 1]) + x[:, 0] * x[:, 1]