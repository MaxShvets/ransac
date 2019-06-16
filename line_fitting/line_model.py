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
        return abs(y - (self._beta0 + x * self._beta1))

    def error(self, points):
        x, y = zip(*list(points))
        return sum(map(lambda xi, yi: self.deviation((xi, yi)), x, y))

    def get_params(self):
        return self._beta0, self._beta1

    def get_points(self):
        return self._points
