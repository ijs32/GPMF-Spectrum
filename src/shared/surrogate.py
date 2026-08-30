from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):


    def __init__(self, x: np.ndarray, func_e: Callable, code: int = 3,  verbose: bool = False):
        self._verbose = verbose
        
        self.func_e = func_e # function to approximate
        self.code   = code # code to pick basis function
        
        self.x = x
        self.y = self.func_e(x)
        
        self._n  = self.x.shape[0]
        self.dim = self.x.shape[1]


    def __call__(self, *coords):
        """
        Helper to make the fitter's model one-to-one
        swappable with the objective target function used to
        build the model. Does not evaluate uncertainty.

        :param coords: tuple of coordinates,
            this should work regardless of dimension of target function.
        """
        x = np.array(coords)
        y = self.evaluate(x)
        return y


    def test_check_model(self):
        """
        A test that checks the model at data points.
        The value should be close to zero.
        :return:
        """
        epss = []
        us = []
        for i in range(self._n):
            if hasattr(self, 'evaluate_uncertainty'):
                ypred, upred = self.evaluate_uncertainty(self.x[i,:])
                us.append(upred)
                
            else:
                ypred = self.__call__(self.x[i,:])
            
            yref = self.y[i]
            eps = abs(ypred - yref)
            epss.append(eps)

        avg = np.mean(epss)
        avg /= self.n
        mx = max(epss)
        mn = min(epss)
        # print(epss)
        print(f"minimum error: {mn}")
        print(f"average error: {avg}")
        print(f"maximum error: {mx}")
        
        if self.code == 8:
            print(f"uncertainties: {us}")
            print(f"avg uncertainty: {np.mean(us)}")
    

    @abstractmethod
    def evaluate(self, x):
        """Return ypred for a given input x."""
