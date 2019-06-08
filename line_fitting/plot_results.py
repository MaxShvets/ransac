import numpy as np
import matplotlib.pyplot as plt


def abline(intercept, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def plot_results(points, fitted_points, model_params):
    for x, y in points:
        plt.scatter(x, y, None, 'r' if (x, y) in fitted_points else 'b')

    abline(*model_params)
    plt.show()
