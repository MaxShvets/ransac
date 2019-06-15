from ransac import ransac_fit, compute_trials_amount
from line_fitting import LineModel, plot_results
import random


def conduct_random_experiment():
    x_bounds = [0, 10]
    n = 100
    slope = random.uniform(-10, 10)
    intercept = random.uniform(-10, 10)
    outlier_probability = 0.4
    tolerance = 0.1
    data = set()
    x = x_bounds[0]
    step = (x_bounds[1] - x_bounds[0]) / n
    min_model_size = 2
    max_trials_amount = compute_trials_amount(1 - outlier_probability, min_model_size)

    def random_error():
        is_error = random.random() < outlier_probability

        if is_error:
            return tolerance * random.uniform(2, 10) * random.choice([-1, 1])
        else:
            return tolerance * random.uniform(-1, 1)

    for i in range(n):
        y = intercept + slope*x + random_error()
        data.add((x, y))
        x += step * random.random()

    fit = ransac_fit(data, LineModel, min_model_size, max_trials_amount, tolerance, 40)
    plot_results(data, fit.get_points(), fit.get_params())
    print("Real params: " + str((intercept, slope)))
    print("Fitted params: " + str(fit.get_params()))


conduct_random_experiment()
