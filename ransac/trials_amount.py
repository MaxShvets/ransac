import math


def compute_trials_amount(inlier_probability, min_model_size):
    p = inlier_probability
    n = min_model_size
    b = p**n

    return math.ceil((2 - p)/b**2 + 3*math.sqrt(1 - b)/b)
