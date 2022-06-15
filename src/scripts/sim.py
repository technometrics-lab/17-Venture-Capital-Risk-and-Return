import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import *


def sim(gamma, delta, sigma, k, a, b, c, d, pim, start_date, logrf, logmk, logv, stockidx, dok, big, sample_size=54):
    def pipo_func(x):
        return 1 / (1 + np.exp(-a * (x - b)))

    assert np.log(k) >= logv[0], 'sim ERROR: k off value grid'
    T = sample_size - start_date - 1
    N = logv.shape[0]

    if big:
        prob_ipo_obs = np.zeros((T, N))
        prob_ipo_hid = np.zeros((T, N))
        prob_bkp_obs = np.zeros((T, N))
        prob_bkp_hid = np.zeros((T, N))
        prob_pvt = np.zeros((T, N))
    else:
        prob_ipo_obs = np.zeros((T, N))
        prob_ipo_hid = np.zeros(T)
        prob_bkp_obs = np.zeros(T)
        prob_bkp_hid = np.zeros(T)
        prob_pvt = np.zeros(T)

    prob_lnV = np.zeros((N, N))

    val_min = logv[0]
    val_max = logv[-1]
    val_step = (val_max - val_min) / (N - 1)

    if start_date == -1 and stockidx > 0:
        prob_lnV = np.zeros(N)
        prob_lnV += (np.exp(-0.5 * (gamma + logrf.mean() + delta * (logmk.mean() - logrf.mean())) ** 2 /
                            (delta ** 2 * logmk.std() ** 2 + sigma ** 2)))
        prob_lnV /= prob_lnV.sum()
    elif stockidx == 0:
        prob_lnV = np.zeros(N)
        prob_lnV = np.exp(-0.5 * gamma ** 2 / sigma ** 2)
        prob_lnV /= prob_lnV.sum()

    if dok:
        prob_bank = (logv <= np.log(k)) * (1 - (np.exp(logv) - np.exp(logv[0])) / (k - np.exp(logv[0])))
    else:
        prob_bank = 0

    min_idx = np.argmin(np.abs(logv), axis=0)
    prob_val = np.zeros(N)
    prob_val[min_idx] = 1e4
    prob_private = prob_val
    prob_exit = pipo_func(logv)

    dlogV = np.arange(val_min - val_max, val_max - val_min + 2*val_step, val_step)

    for t in range(T):
        if start_date >= 0 and stockidx > 0:
            mu = gamma + logrf[start_date+t+1] + delta * (logmk[start_date+t+1] - logrf[start_date+t+1])
            prob_lnV0 = (np.exp(-(dlogV - mu)**2 / (2 * sigma**2)) / (np.exp(dlogV) * sigma * np.sqrt(2 * np.pi)))
            prob_lnV0 = prob_lnV0[:-1] * np.diff(exp(dlogV))
            # OLD AND BAD: prob_lnV0 /= prob_lnV0.sum(), some sort of KDE ??
            
            for i in range(N):
                s = prob_lnV0[N - i - 1: 2 * N - i - 1]
                if prob_lnV0[N - i - 1] > 0 or prob_lnV0[2 * N - i - 2] > 0:
                    prob_lnV[:, i] = s / s.sum()
                else:
                    prob_lnV[:, i] = s

        prob_val = prob_lnV @ prob_private
        phadip = prob_val * prob_exit
        phadbk = prob_val * (1 - prob_exit) * prob_bank
        prob_private = prob_val * (1 - prob_exit) * (1 - prob_bank)

        prob_ipo_obs[t, :] = (1 - pim) * d * phadip + pim * d * phadip.mean()
        if big:
            prob_ipo_hid[t, :] = (1 - d) * phadip
            prob_bkp_obs[t, :] = c * phadbk
            prob_bkp_hid[t, :] = (1 - c) * phadbk
            prob_pvt[t, :] = prob_private
        else:
            prob_ipo_hid[t] = ((1 - d) * phadip).sum()
            prob_bkp_obs[t] = (c * phadbk).sum()
            prob_bkp_hid[t] = ((1 - c) * phadbk).sum()
            prob_pvt[t] = prob_private.sum()

    prob_ipo_obs /= 1e4
    prob_bkp_hid /= 1e4
    prob_bkp_obs /= 1e4
    prob_ipo_hid /= 1e4
    prob_pvt /= 1e4

    probsum = prob_pvt[-1] + prob_ipo_obs.sum() + prob_ipo_hid.sum() + prob_bkp_obs.sum() + prob_bkp_hid.sum();
    assert abs(probsum - 1) < 1e-6, f"sim3 WARNING: Probabilities sum to more or less than one ({probsum:.4f})"
    return prob_pvt, prob_ipo_obs, prob_ipo_hid, prob_bkp_obs, prob_bkp_hid
