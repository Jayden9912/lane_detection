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
            if x != -1:
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
        if num_samples > 10:
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
    y_coords = [
        160.0,
        170.0,
        180.0,
        190.0,
        200.0,
        210.0,
        220.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        300.0,
        310.0,
        320.0,
        330.0,
        340.0,
        350.0,
        360.0,
        370.0,
        380.0,
        390.0,
        400.0,
        410.0,
        420.0,
        430.0,
        440.0,
        450.0,
        460.0,
        470.0,
        480.0,
        490.0,
        500.0,
        510.0,
        520.0,
        530.0,
        540.0,
        550.0,
        560.0,
        570.0,
        580.0,
        590.0,
        600.0,
        610.0,
        620.0,
        630.0,
        640.0,
        650.0,
        660.0,
        670.0,
        680.0,
        690.0,
        700.0,
        710.0,
    ]
    postpro_coords = []
    for lx, ly in predicted_coords:
        tmp_coords = []
        ly = np.array(ly)
        for y in y_coords:
            if y in ly:
                idx = np.where(ly == y)[0][0]
                tmp_coords.append([int(lx[idx]), y])
            else:
                tmp_coords.append([-1, y])
        postpro_coords.append(tmp_coords)
    return postpro_coords
