import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import *
from scipy.stats import lognorm


def sim(gamma, delta, sigma, k, a, b, c, d, pi_err, start, logrf, logmk, logv, stockidx, use_k, big, sample_size=54):
    def pipo_func(x):
        return 1 / (1 + np.exp(-a * (x - b)))
    
    def bkp_prob(x):
        return (x <= np.log(k)) * (1 - (np.exp(x) - np.exp(logv[0])) / (k - np.exp(logv[0])))

    T = sample_size - start - 1
    N = logv.shape[0]
        
    prob_exit_good_data = np.zeros((T, N))
    prob_exit_bad_data = np.zeros(T)
    prob_closed_good_data = np.zeros(T)
    prob_closed_bad_data = np.zeros(T)
    prob_private = np.zeros(T)
    prob_lnV = np.zeros((N, N))

    val_min = logv[0]
    val_max = logv[-1]
    val_step = (val_max - val_min) / (N - 1)

    if stockidx == 0:
        prob_lnV = np.zeros(N)
        prob_lnV = np.exp(-0.5 * gamma**2 / sigma**2)
        prob_lnV /= prob_lnV.sum()

    if use_k:
        prob_bank = bkp_prob(logv)
    else:
        prob_bank = 0

    min_idx = np.argmin(np.abs(logv), axis=0)
    prob_val = np.zeros(N)
    prob_val[min_idx] = 1e4
    prob_pvt = prob_val
    prob_exit = pipo_func(logv)

    dlogV = np.arange(val_min - val_max, val_max - val_min + 2*val_step, val_step)

    for t in range(T):
        if start >= 0 and stockidx > 0:
            mu = gamma + logrf[start+t+1] + delta * (logmk[start+t+1] - logrf[start+t+1])
            distr = lognorm(scale=np.exp(mu), s=sigma)
            prob_lnV0 = np.diff(distr.cdf(np.exp(dlogV)))
            # BAD BUT FAST:
            # prob_lnV0 = np.exp(-(dlogV - mu)**2 / (2 * sigma**2))
            # prob_lnV0 /= prob_lnV0.sum()
            
            
            for i in range(N):
                s = prob_lnV0[N - i - 1: 2 * N - i - 1]
                if prob_lnV0[N - i - 1] > 0 or prob_lnV0[2 * N - i - 2] > 0:
                    prob_lnV[:, i] = s / s.sum()
                else:
                    prob_lnV[:, i] = s

        prob_val = prob_lnV @ prob_pvt
        phadip = prob_val * prob_exit
        phadbk = prob_val * (1 - prob_exit) * prob_bank
        prob_pvt = prob_val * (1 - prob_exit) * (1 - prob_bank)

        prob_exit_good_data[t, :] = (1 - pi_err) * d * phadip + pi_err * d * phadip.mean()
        prob_exit_bad_data[t] = ((1 - d) * phadip).sum()
        prob_closed_good_data[t] = (c * phadbk).sum()
        prob_closed_bad_data[t] = ((1 - c) * phadbk).sum()
        prob_private[t] = prob_pvt.sum()

    prob_exit_good_data /= 1e4
    prob_closed_bad_data /= 1e4
    prob_closed_good_data /= 1e4
    prob_exit_bad_data /= 1e4
    prob_private /= 1e4
    probsum = prob_private[-1] + prob_exit_good_data.sum() + prob_exit_bad_data.sum() + prob_closed_good_data.sum() + prob_closed_bad_data.sum()
    assert abs(probsum - 1) < 1e-6, f"sim ERROR: Probabilities sum to more or less than one ({probsum:.4f})"
    return prob_private, prob_exit_good_data, prob_exit_bad_data, prob_closed_good_data, prob_closed_bad_data
