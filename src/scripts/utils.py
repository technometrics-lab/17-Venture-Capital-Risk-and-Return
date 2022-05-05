import pandas as pd
from numpy import exp, log, array, floor, zeros, abs


def transform_params(gamma, delta, sigma, k, a, b, pim, mask, inv=False):
    sigmin = 1e-4
    mask = array(mask)
    assert any(mask)

    if inv:
        tparams = array([gamma, delta, exp(sigma) + sigmin, exp(k), exp(a), b, 1 / (exp(-pim) + 1)])
        return tparams
    else:
        assert sigma > sigmin, f'transform_params ERROR: cant choose sigma less than {sigmin}'

        tparams = array([gamma, delta, log(sigma - sigmin), log(k), log(a), b, -log(1 / pim - 1)])
        return tparams[mask == True]


def to_decimal_date(date):
    y = floor(date / 1e4).astype(int)
    m = floor(date / 100).astype(int) - 100 * y
    d = date - 1e4 * y - 100 * m
    return y + (m - 1) / 12 + d / 365


def display_return_stats(x):
    fi = sum(x["exit_type"] == 1) / x.shape[0] * 100
    fa = sum(x["exit_type"] == 2) / x.shape[0] * 100
    fb = sum(x["exit_type"] == 3) / x.shape[0] * 100
    freg = sum(x["exit_type"] == 5) / x.shape[0] * 100
    fp = sum(x["exit_type"] == 4) / x.shape[0] * 100
    fsr = sum(x["exit_type"] == 6) / x.shape[0] * 100
    fu = 100 - (fi + fa + fb + freg + fp + fsr)

    print(('Note: following refers to round, not company.\n'
           'Round may end in another round, though company eventually goes public'))
    print(f'\tPercent bankrupt: {fb:.2f}%')
    print(f'\tPercent ipo: {fi:.2f}%')
    print(f'\tPercent acquired: {fa:.2f}%')
    print(f'\tPercent with subsequent round: {fsr:.2f}%')
    print(f'\tPercent Private: {fp:.2f}%')
    print(f'\tPercent Ipo registered: {freg:.2f}%')
    print(f'\tPercent fate unknown: {fu:.2f}%')


def find_case(data, dok, bankhand, start_year=1987):
    cases = zeros(data.shape[0])
    data = data.reset_index()

    for index, row in data.iterrows():
        if (row["exit_type"] in [1, 2, 6]) and (row["exit_date"] != -99) and (row["return_usd"] > 0):
            cases[index] = 1
        elif (row["exit_type"] in [1, 2, 5, 6]) and (row["exit_date"] != -99):
            cases[index] = 2
        elif row["exit_type"] in [1, 2, 5, 6]:
            cases[index] = 3
        elif (row["exit_type"] == 4) or ((row["exit_type"] == 3) and (dok == 0)):
            cases[index] = 4
        elif (row["exit_type"] == 3) and (row["exit_date"] != -99) and (bankhand == 2) and (dok == 1):
            cases[index] = 5.3
        elif (row["exit_type"] == 3) and (row["round_date"] != -99) and (dok == 1):
            cases[index] = 6
        else:
            print(row)
            assert False, f"find_case ERROR: Observation {index} does not fit in a category"
    return cases


def check_series(series1: pd.Series, series2: pd.Series, eps: float = 1e-4) -> None:
    assert series1.size == series2.size, "Series must be of the same size"
    diff = abs(series2.values - series1.values)
    assert (diff < eps).all(), f"Some values differ from more than {eps:.0e}"
