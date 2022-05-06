import numpy as np

from src.scripts.utils import *


def sim(gamma, delta, sigma, k, a, b, c, d, pim, start_date, logrf, logmk, logv, stockidx, dok, big, sample_size=54):
    def pipo_func(x):
        return 1 / (1 + exp(-a * (x - b)))

    assert np.log(k) >= logv[0], 'sim ERROR: k off value grid'
    T = sample_size - start_date
    N = logv.shape[0]
    filler = np.ones(N)

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

    pmat = np.zeros((N, N))

    val_bot = logv[0]
    val_rng = logv[-1]
    val_step = (val_rng - val_bot) / (N - 1)

    if start_date == 0 and stockidx > 0:
        pmat = (np.exp(-0.5 * (logv * filler.T - filler * logv.T - (
                    gamma + logrf.mean() + delta * (logmk.mean() - logrf.mean()))) ** 2
                       / (delta ** 2 * logmk.std() ** 2 + sigma ** 2)))
        pmat /= pmat.sum()
    elif stockidx == 0:
        pmat = np.exp(-0.5 * (logv * filler.T - filler * logv.T - gamma) ** 2 / sigma ** 2)
        pmat /= pmat.sum()

    if dok:
        prob_bank = (logv <= np.log(k)) * (1 - (np.exp(logv) - np.exp(logv[0])) / (k - np.exp(logv[0])))
    else:
        prob_bank = 0

    prob_ipo = pipo_func(logv)
    prob_val = np.zeros(N)

    min_idx = np.argmin(np.abs(logv), axis=0)
    prob_val[min_idx] = 1e4
    prob_private = prob_val

    for t in range(T):
        if start_date > 0 and stockidx > 0:
            pmat0 = np.arange(val_bot - val_rng, val_rng - val_bot + val_step, val_step)
            pmat0 = np.exp(-0.5 * (pmat0 - (gamma + logrf[start_date + t] + delta * (
                        logmk[start_date + t] - logrf[start_date + t]))) ** 2 / sigma ** 2)
            pmat0 /= pmat0.sum()

            for i in range(N):
                s = pmat0[N - i - 1: 2 * N - i - 1]
                if pmat0[N - i - 1] > 0 or pmat0[2 * N - i - 2] > 0:
                    pmat[:, i] = s / s.sum()
                else:
                    pmat[:, i] = s

        prob_val = pmat @ prob_private
        phadip = np.multiply(prob_val, prob_ipo)
        phadbk = np.multiply(np.multiply(prob_val, 1 - prob_ipo), prob_bank)
        prob_private = np.multiply(np.multiply(prob_val, 1 - prob_ipo), 1 - prob_bank)

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
