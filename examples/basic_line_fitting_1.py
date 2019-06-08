from ransac import ransac_fit
from line_fitting import LineModel
import numpy as np
import matplotlib.pyplot as plt


def abline(intercept, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


if __name__ == "__main__":
    data = {(0, 0), (1, 1), (2, 2), (3.5, 2), (3, 3), (4, 4), (10, 2)}
    fit = ransac_fit(data, LineModel, 2, 21, 0.01, 4)

    print("Params: ")
    print(fit.get_params())
    print("Points: ")
    print(fit.get_points())

    for x, y in data:
        plt.scatter(x, y, None, 'r' if (x, y) in fit.get_points() else 'b')

    abline(*fit.get_params())
    plt.show()
