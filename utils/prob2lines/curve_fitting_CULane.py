import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor


class PolynomialRegression(object):
    def __init__(self, degree=4, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {"coeffs": self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def lane_coords_preprocess(lane_coords):
    filtered_coords = []
    for l in lane_coords:
        tmp_x = []
        tmp_y = []
        tmp_l = []
        for (x, y) in l:
            tmp_x.append(x)
            tmp_y.append(y)
        tmp_l.append(tmp_x)
        tmp_l.append(tmp_y)
        filtered_coords.append(tmp_l)
    return filtered_coords


def ransac_fitting(filtered_lane_coords, poly_degree=3):
    predicted_coords = []
    for l in filtered_lane_coords:
        tmp_coords = []
        x_val = np.array(l[0])
        y_val = np.array(l[1])

        assert len(x_val) == len(y_val)

        num_samples = len(x_val)
        if num_samples > 5:
            ransac = RANSACRegressor(
                PolynomialRegression(degree=poly_degree),
                residual_threshold=0.1 * np.std(x_val),
                min_samples=int(0.4 * num_samples),
                max_trials=100,
                random_state=0,
            )
            ransac.fit(np.expand_dims(y_val, axis=1), x_val)
            y_min = np.min(y_val)
            y_max = np.max(y_val)
            y_val = []
            for i in range(y_min, y_max + 1, 10):
                y_val.append(i)
            x_hat = ransac.predict(np.expand_dims(y_val, axis=1))
            tmp_coords.append(x_hat)
            tmp_coords.append(y_val)
        else:
            tmp_coords.append(x_val)
            tmp_coords.append(y_val)
        predicted_coords.append(tmp_coords)
    return predicted_coords


def lane_coords_postprocess(predicted_coords):
    y_coords = [99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589]
    postpro_coords = []
    for lx, ly in predicted_coords:
        tmp_coords = []
        ly = np.array(ly)
        for y in y_coords:
            if y in ly:
                idx = np.where(ly == y)[0][0]
                tmp_coords.append([int(lx[idx]), y])
        postpro_coords.append(tmp_coords)
    return postpro_coords