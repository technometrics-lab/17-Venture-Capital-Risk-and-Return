from numpy import diag, array, sqrt
from pandas import DataFrame
from scipy.optimize import minimize

from scripts.findlkh import find_likelyhood
from scripts.utils import transform_params


class Model:
    def __init__(self, x, xc, logrf, logmk, minage, c, d, logv, mask, stockidx, dok, start_year, sample_size):
        self.x = x.to_numpy()
        self.logrf = logrf.to_numpy()
        self.logmk = logmk.to_numpy()
        self.logv = logv
        self.xc = xc
        self.minage = minage
        self.c = c
        self.d = d
        self.mask = mask
        self.stockidx = stockidx
        self.dok = dok
        self.start_year = start_year
        self.sample_size = sample_size
        self.curr_iter = 0

    def model_likelyhood(self, tpars):
        return find_likelyhood(tpars, self.x, self.xc, self.logrf, self.logmk, self.minage,
                               self.c, self.d, self.logv, self.mask, self.stockidx,
                               self.dok, self.start_year, self.sample_size)

    def optimize_likelyhood(self, tpar0, mask, maxiter=15):
        self.curr_iter = 0
        print("{0:<5}{1:<10}{2:<10}{3:<10}{4:<10}{5:<10}{6:<10}{7:<10}{8:<10}"
              .format("iter", "gamma", "delta", "sigma", "k", "a", "b", "pim", "lkh"))
        res = minimize(self.model_likelyhood, tpar0, options={'maxiter': maxiter}, callback=self.printer_callback)

        resx, resh = res.x, res.hess_inv
        params = transform_params(*resx, mask, inv=True)
        labels = array(['gamma', 'delta', 'sigma', 'k', 'a', 'b', 'pim'])[mask]

        std = sqrt(diag(resh))[mask]  # fisher information matrix
        res = list(map(list, zip(params, std)))
        res = DataFrame(dict(zip(labels, res)), index=['value', 'std'])
        return res

    def printer_callback(self, x):
        print((f'{self.curr_iter:<5}{x[0]:<10.4f}{x[1]:<10.4f}{x[2]:<10.4f}{x[3]:<10.4f}'
               f'{x[4]:<10.4f}{x[5]:<10.4f}{x[6]:<10.4f}{self.model_likelyhood(x):<10.4f}'))
        self.curr_iter += 1
