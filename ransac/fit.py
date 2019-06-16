import random


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

        model_with_inliers = model_constructor(model_points | inliers)

        if min_accepted_size <= len(inliers) + min_model_size:
            return model_with_inliers

        error = model_with_inliers.error(data)

        if (not smallest_error) or error < smallest_error:
            smallest_error = error
            best_model = model_with_inliers

        iterations_num += 1

    return best_model
