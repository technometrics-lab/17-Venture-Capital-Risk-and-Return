import numpy as np

from src.scripts.market_api import get_yfinance_time_series, get_fred_time_series
from src.scripts.utils import *


def load_index_data(source='cochrane', ticker: str = None, start: str = None, end: str = None, freq: str = None):
    if source == 'cochrane':
        data = pd.read_csv('../../data/cochrane_data/sptr.csv')
        idx = data['data']
    else:
        assert ticker is not None, "Ticker must be specified when fetching from API"
        idx = get_yfinance_time_series(ticker, start, end, freq)

    logspret = np.log(idx.pct_change().dropna() + 1)
    return logspret


def load_tbills_data(source='cochrane', series_id: str = None, start: str = None, end: str = None, freq: str = None):
    if source == 'cochrane':
        data = pd.read_csv('../../data/cochrane_data/tb3ms.csv')
        rf = data['rf']
    else:
        assert series_id is not None, "Series id must be specified when fetching from API"
        rf = get_fred_time_series(series_id, start, end)

    logrf = np.log(1 + rf[::3] / 400)[:-1]
    return logrf


def load_venture_data(roundret, round_code, industry_code, test=False):
    if not test:
        data = pd.read_csv('../../data/cochrane_data/returns.csv')
        end_year = 2000
    else:
        data = pd.read_csv('../data/vc_data.csv')
        end_year = 2022

    if (roundret == 1) or (round_code > 0):
        num_rounds = data.groupby("company_num").size()
        round_idx = np.concatenate(num_rounds.apply(lambda x: np.arange(x)).ravel())
        data["round_num"] = round_idx + 1

        data.groupby("company_num").apply(handle_group)

        if round_code > 0:
            data = data[data["round_num"] == round_code]
            assert not data.empty, f"No data points for rounds = {round_code}"

    if industry_code > 0:
        data = data[data.group_num == industry_code]
        assert not data.empty, f"No data points for industry = {industry_code}"

    # set return to 0 for out of business projects
    data["return_usd"] *= (data["exit_type"] != 3)
    # keep only projects with know exit type
    data = data[data["exit_type"] != -99]
    # keep only projects with correct ownership data
    data = data[data["flagpct"] != 1]
    # keep only projects with correct round dates
    data = data[data["round_date"] != -99]
    # discard round dates after first quarter of 2000 (need lag for returns)
    data = data[to_decimal_date(data["round_date"]) < end_year + 0.25]

    # set missing exit date when value is before existing round date
    wrong_date_idx = (data["exit_date"] != -99) & (
                to_decimal_date(data["exit_date"]) <= to_decimal_date(data["round_date"]))
    data.loc[wrong_date_idx, "exit_date"] = -99
    # set state to out of business when last return is 0
    wrong_ret_idx = (data["return_usd"] == 0) & (data["exit_type"] != 3)
    data.loc[wrong_ret_idx, "exit_type"] = 3

    # discard super-extreme returns: hand-checked incorrect values
    data = data[data["return_usd"] <= 300]

    return data.reset_index(drop=True)


def handle_group(group):
    group["exit_date"] = group.round_date.shift(-1, fill_value=-99)
    group["exit_value"] = group.postvalue_usd.shift(-1, fill_value=-99)
    group["exit_type"][:-1] = 6

    pmv_start = group.postvalue_usd
    pmv_end = group.postvalue_usd.shift(-1, fill_value=-99)
    raised = group.raised_usd.shift(-1, fill_value=-99)

    flag1 = (pmv_end > 0) & (raised > 0) & (pmv_start > 0)
    flag2 = (pmv_end > 0) & (raised > 0) & (pmv_end == raised) & ~flag1
    flag0 = ~flag1 & ~flag2
    group["return_usd"] = (pmv_end - raised) / pmv_start * flag1 + flag0 * (-99)
    return group
