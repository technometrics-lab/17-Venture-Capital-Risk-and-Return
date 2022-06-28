import pandas as pd
import pickle
from scripts.preprocess import *
from scripts.minimize_lkh import Model
from scripts.utils import display_return_stats, find_case, transform_params


def get_dates(x, test=False):
    start_date, end_date = min(x.round_date), max(x.round_date)
    if test:
        s, e = str(start_date), str(end_date)
        start_date = pd.to_datetime(f"{s[:4]}-{s[4:6]}-{s[-2:]}")
        end_date = pd.to_datetime(f"{e[:4]}-{e[4:6]}-{e[-2:]}")
    return start_date, end_date


def main(gamma, delta, sigma, k, a, b, pi, impose_alpha=False, stockidx=1, nopi=0, use_k=1, bankhand=2, test=False, pred=True):
    x = load_venture_data(test=test, pred=pred)
    start_date, end_date = get_dates(x, test)
    size = (end_date.to_period(freq='Q') - start_date.to_period(freq='Q')).n + 2
    start, end = start_date.year, end_date.year
    if test:
        start = '1987-01-01'
        end = '2000-08-02'

    logrf = load_tbills_data('TB3MS', start, end)
    logmk = load_index_data('^SP500TR', start, end, '1mo', test)

    print(f"number of observations: {x.shape[0]}")
    x["ddate"] = to_decimal_date(x["round_date"])
    x = x.sort_values(by=["ddate", "company_num"]).drop(columns=["ddate"])

    display_return_stats(x)

    c = sum((x["exit_type"] == 3) & (x["exit_date"] != -99)) / sum((x["exit_type"] == 3))
    print(f'\tPercent of bankrupt have good data. Using this parameter in simulation: {c * 100:.2f}%')

    good_exit = x["exit_type"].isin([1, 2, 5, 6])
    good_date = x["exit_date"] != -99
    good_return = x["return_usd"] > 0
    d = (good_exit & good_date & good_return).sum() / good_exit.sum()
    print(f'\tPercent of valuations (ipo, acquired, new round) that have good data: {d * 100:.2f}%\n\n')

    minage = 0.25
    logv = np.arange(-7, 7.1, 0.1)
    pi = 0 if nopi == 1 else pi
    xc = find_case(x, use_k, bankhand)
    mask = [impose_alpha != 1, stockidx > 0, True, use_k != 0, True, True, nopi != 1]
    tpar0 = transform_params(gamma, delta, sigma, k, a, b, pi, mask)

    model = Model(x, xc, logrf, logmk, minage, c, d, logv, mask, stockidx, use_k, start_year=to_decimal_date(start_date), sample_size=size)
    model.model_likelyhood(tpar0)
    return model.optimize_likelyhood(tpar0, mask, maxiter=30)


if __name__ == "__main__":
    gamma0 = 0.01
    delta0 = 1.5
    sigma0 = 0.9
    k0 = 0.1
    a0 = 1
    b0 = 3
    pi0 = 0.01

    res = main(gamma0, delta0, sigma0, k0, a0, b0, pi0, test=False, pred=True)
    
    print(res)
    with open('res.pkl', 'wb') as file:
        pickle.dump(res, file)
        

###### TODO #######
# run simulations for all datasets: predicitons, cochrane, no pred, bootstrap, 
# same but run by sector
# add missing exit values (go with 1st day market cap, fk it)
# improve selection function
# draw graphs of empiracal CDFs