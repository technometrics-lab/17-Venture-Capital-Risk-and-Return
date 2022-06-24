import numpy as np
import math
from scripts.utils import *
from scripts.sim import sim


def find_likelyhood(tpars, x, xc, logrf, logmk, minage, c, d, logv, mask,
                    stockidx, use_k, start_year=1987, sample_size=54):

    xlo = transform_params(-1, -10, 0.1, 0.01, 0.1, 0.1, 1e-6, mask)
    xhi = transform_params(1, 10, 5, 1.5, 10, 10, 1 - 1e-6, mask)

    penalty = 1e6 * sum((tpars < xlo) * (xlo - tpars) + (tpars > xhi) * (tpars - xhi))**2
    tpars = (tpars < xlo) * xlo + (tpars > xhi) * xhi + ((tpars >= xlo) & (tpars <= xhi)) * tpars
    gamma, delta, sigma, k, a, b, pi_err = transform_params(*tpars, mask, inv=True)

    if not mask[6]:
        pi_err = 0

    if not mask[0]:
        if not stockidx:
            gamma = log(1 + 15 / 400) - 0.5 * sigma**2
        elif stockidx == 1:
            mlogrf = logrf.mean()
            mlogmk = logmk.mean()
            msigma = sigma.mean()
            gamma = (-log(exp(delta * (mlogmk - mlogrf) + 0.5 * (delta * msigma)**2 + 0.5 * sigma**2)
                          * (1 + (exp(delta * msigma**2) - 1) * (exp(-(mlogmk - mlogrf) - 0.5 * msigma**2) - 1)
                             / (exp(msigma**2) - 1))))
        else:
            assert False, 'find_likelyhood ERROR: trying to impose alpha = 0 for model other than SP500 or no stock.'

    if (log(k) <= logv.min()) | (log(k) > logv.max()):
        print(('find_likelyhood being asked for k < min of grid or > max of grid '
               '-- sending back huge likly. SHOULD NOT SEE THIS.'))
        print('better to raise penalty in find_likelyhood so that the likelyhood function is continuous')
        return np.inf

    quarter_index = -1
    sim_params = [gamma, delta, sigma, k, a, b, c, d, pi_err, quarter_index, logrf, logmk, logv, stockidx, use_k, 0, sample_size]
    if not stockidx:
        prob_pvt, prob_ipo_obs, prob_ipo_hid, prob_bkp_obs, prob_bkp_hid = sim(*sim_params)

    lk = 0
    for il in range(x.shape[0]):
        newprob = -1
        round_date = to_decimal_date(x[il, 3])
        exit_date = to_decimal_date(x[il, 6]) if x[il, 6] != -99 else x[il, 6]
        round_index = math.floor((round_date - start_year) * 4) # should be the quarter in which this project starts
        
        if round_index != quarter_index: # if this observation has a new begin date, do sim again
            if stockidx > 0:  # version with stock index
                sim_params[9] = round_index
                prob_pvt, prob_ipo_obs, prob_ipo_hid, prob_bkp_obs, prob_bkp_hid = sim(*sim_params)
            quarter_index = round_index

        match xc[il]:
            # ipo/acq with good date and return
            case 1:
                # quarter of exit with 1987:1 = 1 - start
                exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
                logV = log(x[il, 9])
                logVindx = np.argmin(np.abs(logV - logv))  # find nearest value gridpoint
                if exit_index + 1 <= minage * 4:  # treat dates less than minage as "on or before"
                    newprob = sum(prob_ipo_obs[:int(min(minage * 4, prob_ipo_obs.shape[0])), logVindx])
                else:
                    newprob = prob_ipo_obs[exit_index, logVindx]

            # ipo/acq/another round with bad return but good dates
            case 2:
                exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
                if exit_index + 1 < minage * 4:
                    newprob = sum(prob_ipo_hid[:int(min(minage * 4, prob_ipo_hid.shape[0]))])
                else:
                    newprob = prob_ipo_hid[exit_index]

            # ipo/acq with bad return and bad end date (unused)
            case 3:
                exit_index = sample_size - quarter_index - 2
                newprob = sum(prob_ipo_hid[:exit_index + 1])

            # still private, good dates
            case 4:
                exit_index = prob_pvt.shape[0] - 1
                newprob = prob_pvt[exit_index]

            # use dates to infer bankrupt 'on or before' given date */
            case 5.3:
                exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
                if exit_index + 1 > minage * 4:
                    newprob = sum(prob_bkp_obs[:exit_index + 1])
                elif sum(prob_bkp_obs[:int(min(minage * 4, prob_bkp_obs.shape[0]))]) > 0:
                    newprob = prob_bkp_obs[exit_index]
                        
            # bankrupt, dates not ok
            case 6:
                exit_index = prob_pvt.shape[0] - 1
                newprob = sum(prob_bkp_hid[:exit_index + 1])
        
        if newprob > 0:
            lk = lk + log(newprob)
        else:
            return np.inf

    lk = - lk + penalty
    return lk
