
import GPy
import numpy as np


class EmuKitGP():

    def __init__(self, x, y, xmax, xmin):

        self.x = x
        self.y = np.asarray(y, dtype=float).reshape(-1, 1)

        self.dim = x.shape[1]
        self.xmin = xmin
        self.xmax = xmax

        self.fit_model()


    def fit_model(self):
        kernel = GPy.kern.Matern52(self.dim, ARD=True)
        for d, (lo, hi) in enumerate(zip(self.xmin, self.xmax)):
            kernel.lengthscale[[d]].constrain_bounded(1e-3, hi - lo)
        kernel.variance.constrain_bounded(1e-3, 1e3)

        self.model = GPy.models.GPRegression(self.x, self.y, kernel)

        self.model.likelihood.variance.constrain_bounded(1e-6, 1e-1)
        self.model.likelihood.variance[:] = 1e-2

        self.model.optimize_restarts(num_restarts=15)


    def evaluate(self, x):
        mean, _ = self.model.predict(x)
        return mean


    def evaluate_uncertainty(self, x):
        mean, var = self.model.predict(np.atleast_2d(x))
        return mean, np.sqrt(var)