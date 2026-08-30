from collections.abc import Callable

import GPy
import numpy as np
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays


class EmukitMFGP:
    def __init__(self, x_lo, y_lo, x_hi, y_hi, xmin, xmax):
        self.dim = x_hi.shape[1]
        self.xmin, self.xmax = xmin, xmax

        y_lo = np.asarray(y_lo, float).reshape(-1, 1)
        y_hi = np.asarray(y_hi, float).reshape(-1, 1)

        self.X_train, self.Y_train = convert_xy_lists_to_arrays([x_lo, x_hi], [y_lo, y_hi])
        self.fit_model()


    def fit_model(self):
        kernels = [GPy.kern.Matern52(self.dim, ARD=True) for _ in range(2)]
        for k in kernels:
            for d, (lo, hi) in enumerate(zip(self.xmin, self.xmax)):
                k.lengthscale[[d]].constrain_bounded(1e-3, 10 * (hi - lo))
            k.variance.constrain_bounded(1e-3, 1e3)

        mf_kernel = LinearMultiFidelityKernel(kernels)
        self.model = GPyLinearMultiFidelityModel(
            self.X_train, self.Y_train, mf_kernel, n_fidelities=2)

        # fidelity 0 — generated, essentially noiseless
        self.model.mixed_noise.Gaussian_noise.variance[:] = 1e-6
        self.model.mixed_noise.Gaussian_noise.variance.fix()

        # fidelity 1 — PAMELA, let the data set the level
        self.model.mixed_noise.Gaussian_noise_1.variance.constrain_bounded(1e-6, 1e-1)
        self.model.mixed_noise.Gaussian_noise_1.variance[:] = 1e-2

        self.model.optimize_restarts(num_restarts=5)

    def _as_high(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        col = np.ones((x.shape[0], 1))
        return np.hstack([x, col]), {'output_index': col.astype(int)}

    def evaluate(self, x):
        Xq, meta = self._as_high(x)
        mean, _ = self.model.predict(Xq, Y_metadata=meta)
        return mean

    def evaluate_uncertainty(self, x):
        Xq, meta = self._as_high(x)
        mean, var = self.model.predict(Xq, Y_metadata=meta)
        return mean, np.sqrt(var)