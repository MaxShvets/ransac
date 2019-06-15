import random
import math


def compute_trials_amount(inlier_probability, min_model_size):
    p = inlier_probability
    n = min_model_size
    b = p**n

    return math.ceil((2 - p)/b**2 + 3*math.sqrt(1 - b)/b)


def ransac_fit(data, model_constructor, min_model_size, max_iterations_num, error_tolerance, min_accepted_size):
    iterations_num = 0
    best_model = None
    smallest_error = None

    while iterations_num < max_iterations_num:
        model_points = set(random.sample(data, min_model_size))
        model = model_constructor(model_points)
        inliers = set()

        for point in data - model_points:
            if model.deviation(point) < error_tolerance:
                inliers.add(point)

        if min_accepted_size <= len(inliers) + min_model_size:
            model_with_inliers = model_constructor(model_points | inliers)
            error = model_with_inliers.error()

            if (not smallest_error) or error < smallest_error:
                smallest_error = error
                best_model = model_with_inliers

        iterations_num += 1

    return best_model
