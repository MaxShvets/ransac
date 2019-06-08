from ransac import ransac_fit
import numpy as np
import matplotlib.pyplot as plt


def abline(intercept, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


class LineModel:
    def __init__(self, points):
        self._points = points
        x, y = zip(*list(points))
        n = len(x)
        self._x = x
        self._y = y
        y_sum = sum(y)
        x_sum = sum(x)
        x_sq_sum = sum(map(lambda xi: xi**2, x))
        multiple_sum = sum(map(lambda xi, yi: xi*yi, x, y))
        self._beta1 = (n * multiple_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum ** 2)
        self._beta0 = (1 / n) * y_sum - (self._beta1 / n) * x_sum

    def deviation(self, point):
        x, y = point
        return (y - (self._beta0 + x * self._beta1)) ** 2

    def error(self):
        return sum(map(lambda xi, yi: self.deviation((xi, yi)), self._x, self._y))

    def get_params(self):
        return self._beta0, self._beta1

    def get_points(self):
        return self._points


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
