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
        elif stockidx is not None:
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

        if xc[il] == 1:
            newprob = good_return_and_date(x, minage, logv, start_year, quarter_index, prob_ipo_obs, il, exit_date)
        elif xc[il] == 2:
            newprob = good_date_bad_return(minage, start_year, quarter_index, prob_ipo_hid, exit_date)
        elif xc[il] == 3:
            newprob = bad_return_and_date(sample_size, quarter_index, prob_ipo_hid)
        elif xc[il] == 4:
            newprob = still_private(prob_pvt)
        elif xc[il] == 5.3:
            newprob = bankrupt_and_good_dates(minage, start_year, quarter_index, prob_bkp_obs, exit_date)      
        elif xc[il] == 6:
            newprob = bankrupt_and_bad_dates(prob_pvt, prob_bkp_hid)
        
        if newprob > 0:
            lk = lk + log(newprob)
        else:
            return np.inf

    lk = - lk + penalty
    return lk


def bankrupt_and_bad_dates(prob_pvt, prob_bkp_hid):
    exit_index = prob_pvt.shape[0] - 1
    newprob = sum(prob_bkp_hid[:exit_index + 1])
    return newprob

def bankrupt_and_good_dates(minage, start_year, quarter_index, prob_bkp_obs, exit_date):
    exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
    if exit_index + 1 > minage * 4:
        newprob = sum(prob_bkp_obs[:exit_index + 1])
    else:
        newprob = prob_bkp_obs[exit_index]
    return newprob

def still_private(prob_pvt):
    exit_index = prob_pvt.shape[0] - 1
    newprob = prob_pvt[exit_index]
    return newprob

def bad_return_and_date(sample_size, quarter_index, prob_ipo_hid):
    exit_index = sample_size - quarter_index - 2
    newprob = sum(prob_ipo_hid[:exit_index + 1])
    return newprob

def good_date_bad_return(minage, start_year, quarter_index, prob_ipo_hid, exit_date):
    exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
    if exit_index + 1 < minage * 4:
        newprob = sum(prob_ipo_hid[:int(min(minage * 4, prob_ipo_hid.shape[0]))])
    else:
        newprob = prob_ipo_hid[exit_index]
    return newprob

def good_return_and_date(x, minage, logv, start_year, quarter_index, prob_ipo_obs, il, exit_date):
    # quarter of exit with 1987:1 = 1 - start
    exit_index = math.floor((exit_date - start_year) * 4) - quarter_index - 1
    logV = log(x[il, 9])
    logVindx = np.argmin(np.abs(logV - logv))  # find nearest value gridpoint
    if exit_index + 1 <= minage * 4:  # treat dates less than minage as "on or before"
        newprob = sum(prob_ipo_obs[:int(min(minage * 4, prob_ipo_obs.shape[0])), logVindx])
    else:
        newprob = prob_ipo_obs[exit_index, logVindx]
    return newprob
