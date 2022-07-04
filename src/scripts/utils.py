import pandas as pd
from tqdm import tqdm
from numpy import exp, log, array, floor, zeros, abs, int64, float64, sqrt


def transform_params(gamma, delta, sigma, k, a, b, pi_err, mask, inv=False):
    sigmin = 1e-4
    mask = array(mask)
    assert any(mask)

    if inv:
        tparams = array([gamma, delta, exp(sigma) + sigmin, exp(k), exp(a), b, 1 / (exp(-pi_err) + 1)])
        return tparams
    else:
        assert sigma > sigmin, f'transform_params ERROR: cant choose sigma less than {sigmin}'

        tparams = array([gamma, delta, log(sigma - sigmin), log(k), log(a), b, -log(1 / pi_err - 1)])
        return tparams[mask]


def to_decimal_date(date):
    if isinstance(date, pd.Series):
        if isinstance(date.loc[0], int64) or isinstance(date.loc[0], float64):
            y = floor(date / 1e4).astype(int)
            m = floor(date / 100).astype(int) - 100 * y
            d = date - 1e4 * y - 100 * m
        else:
            y = date.apply(lambda x: x.year)
            m = date.apply(lambda x: x.month)
            d = date.apply(lambda x: x.day)
    else:
        if isinstance(date, int64) or isinstance(date, float64):
            y = floor(date / 1e4).astype(int)
            m = floor(date / 100).astype(int) - 100 * y
            d = date - 1e4 * y - 100 * m
        else:
            y = date.year
            m = date.month
            d = date.day
    return y + (m - 1) / 12 + (d - 1) / 365


def display_return_stats(x):
    fi = sum(x["exit_type"] == 1) / x.shape[0] * 100
    fa = sum(x["exit_type"] == 2) / x.shape[0] * 100
    fb = sum(x["exit_type"] == 3) / x.shape[0] * 100
    freg = sum(x["exit_type"] == 5) / x.shape[0] * 100
    fp = sum(x["exit_type"] == 4) / x.shape[0] * 100
    fsr = sum(x["exit_type"] == 6) / x.shape[0] * 100
    fu = 100 - (fi + fa + fb + freg + fp + fsr)

    print(f"number of observations: {x.shape[0]}")
    print(('Note: following refers to round, not company.\n'
           'Round may end in another round, though company eventually goes public'))
    print(f'\tPercent bankrupt: {fb:.2f}%')
    print(f'\tPercent ipo: {fi:.2f}%')
    print(f'\tPercent acquired: {fa:.2f}%')
    print(f'\tPercent with subsequent round: {fsr:.2f}%')
    print(f'\tPercent Private: {fp:.2f}%')
    print(f'\tPercent Ipo registered: {freg:.2f}%')
    print(f'\tPercent fate unknown: {fu:.2f}%')


def find_case(data, use_k, bankhand):
    cases = zeros(data.shape[0])
    data = data.reset_index()
    print("Finding category for each observation")
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        if (row["exit_type"] in [1, 2, 6]) and (row["exit_date"] != -99) and (row["return_usd"] > 0):
            cases[index] = 1
        elif (row["exit_type"] in [1, 2, 5, 6]) and (row["exit_date"] != -99):
            cases[index] = 2
        elif row["exit_type"] in [1, 2, 5, 6]:
            cases[index] = 3
        elif (row["exit_type"] == 4) or ((row["exit_type"] == 3) and (use_k == 0)):
            cases[index] = 4
        elif (row["exit_type"] == 3) and (row["exit_date"] != -99) and (bankhand == 2) and (use_k == 1):
            cases[index] = 5.3
        elif (row["exit_type"] == 3) and (row["round_date"] != -99) and (use_k == 1):
            cases[index] = 6
        else:
            print(row)
            assert False, f"find_case ERROR: Observation {index} does not fit in a category"
    return cases


def check_series(series1: pd.Series, series2: pd.Series, eps: float = 1e-4) -> None:
    assert series1.size == series2.size, "Series must be of the same size"
    diff = abs(series2.values - series1.values)
    assert (diff < eps).all(), f"Some values differ from more than {eps:.0e}"


def get_beta(gamma, delta, sigma, log_mk, log_rf):
    return exp(gamma + (delta - 1) * (log_mk.mean() - log_rf.mean())
               + 1 / 2 * sigma ** 2 + 1 / 2 * (delta ** 2 - 1)
               * log_mk.std() ** 2) * (exp(delta * log_mk.std() ** 2)
            - 1) / (exp(log_mk.std() ** 2) - 1)


def get_alpha(gamma, delta, sigma, log_mk, log_rf, beta):
    return exp(log_rf.mean()) * (exp(gamma + delta * (log_mk.mean()
                                 - log_rf.mean()) + 1 / 2 * delta ** 2
                                 * log_mk.std() ** 2 + 1 / 2 * sigma
                                 ** 2) - 1 - beta * (exp(log_mk.mean()
                                 - log_rf.mean() + 1 / 2 * log_mk.std()
                                 ** 2) - 1))
    
def print_results(results, log_mk, log_rf):
    mu_mk = log_mk.mean()
    sg_mk = log_mk.std()
    mu_rf = log_rf.mean()
    gamma, delta, sigma, k, a, b, pi = results.loc['value']
    sdg, sdd, sds, sdk, sda, sdb, sdpi = results.loc['std']
    print('Using parameters (annualized percentages)')
    print(f'E[log Rf]={400*mu_rf:.2f}%, E[log Rm]={400*mu_mk:.2f}%, V[log Rm]={200*sg_mk:.2f}%')
    
    # mean and sd of quarterly log returns
    Elnr = gamma + mu_rf + delta * (mu_mk - mu_rf)
    siglnr = sqrt(delta**2 * sg_mk**2 + sigma**2)
    
    er = 400*(exp(Elnr+1/2*siglnr**2)-1);
    sdr = 200*sqrt(((er/400+1)*(exp(siglnr**2)-1)))
    
    beta = get_beta(*results.loc['value'][:3], log_mk, log_rf)
    alpha = get_alpha(*results.loc['value'][:3], log_mk, log_rf, beta)
    
    implied = pd.DataFrame({'E[ln R] (%)':400*Elnr, 'SD[ln R] (%)': 200*siglnr, 'E[R] (%)': er, 'SD[R] (%)': sdr, 'alpha (%)': 400*alpha, 'beta':beta}, index=['value'])
    params  = pd.DataFrame({
        'gamma (%)': [400 * gamma, 400 * sdg],
        'delta': [delta,  sdd],
        'sigma (%)': [200 * sigma, 200 * sds],
        'k (%)': [100 * k, 100 * sdk * k],
        'a': [a, a * sda],
        'b': [b, sdb],
        'pi (%)': [100 * pi, 100 * sdpi * pi * (1 - pi)]
    }, index=['value', 'sd'])
    
    return implied, params