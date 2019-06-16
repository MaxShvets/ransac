from ransac import ransac_fit, compute_trials_amount
from line_fitting import LineModel, plot_results
import random
import numpy as np


def conduct_random_experiment():
    x_bounds = [0, 10]
    n = 100
    slope = random.uniform(-10, 10)
    intercept = random.uniform(-10, 10)
    outlier_probability = 0.4
    tolerance = 0.1
    points = set()
    x = x_bounds[0]
    step = (x_bounds[1] - x_bounds[0]) / n
    min_model_size = 2
    inlier_probability = 1 - outlier_probability
    max_trials_amount = compute_trials_amount(
        inlier_probability, min_model_size
    )
    min_accepted_size = n * inlier_probability - 20

    def random_error():
        is_error = random.random() < outlier_probability

        if is_error:
            return (
                tolerance
                * random.uniform(2, 20)
                * random.choice([-1, 1])
            )
        else:
            return tolerance * random.uniform(-1, 1)

    for i in range(n):
        y = intercept + slope*x + random_error()
        points.add((x, y))
        x += step * random.random()

    fit = ransac_fit(
        points, LineModel, min_model_size,
        max_trials_amount, tolerance, min_accepted_size
    )

    return {
        "fit": fit,
        "real_params": (intercept, slope),
        "points": points,
        "fitting_model_found":
            min_accepted_size <= len(fit.get_points())
    }


def calculate_fit_wellness(result):
    real_intercept, real_slope = result["real_params"]
    fitted_intercept, fitted_slope = result["fit"].get_params()

    return (
        (fitted_intercept - real_intercept)**2
        + (fitted_slope - fitted_slope)**2
    )


def conduct_random_experiment_series():
    results = []

    for i in range(1000):
        result = conduct_random_experiment()

        if result["fitting_model_found"]:
            results.append(result)

    print("Successful experiments: ", len(results))
    worst_fit_result_index = np.argmax(
        map(calculate_fit_wellness, results)
    )
    worst_fit_result = results[worst_fit_result_index]
    worst_fit = worst_fit_result["fit"]

    print("Worst fit real params: ", worst_fit_result["real_params"])
    print("Worst fit fitted params: ", worst_fit.get_params())
    plot_results(
        worst_fit_result["points"],
        worst_fit.get_points(),
        worst_fit.get_params()
    )


conduct_random_experiment_series()
