from ransac import ransac_fit
from line_fitting import LineModel, plot_results


data = {(0, 0), (1, 1), (2, 2), (3.5, 2), (3, 3), (4, 4), (10, 2)}
fit = ransac_fit(data, LineModel, 2, 21, 0.01, 4)
plot_results(data, fit.get_points(), fit.get_params())
