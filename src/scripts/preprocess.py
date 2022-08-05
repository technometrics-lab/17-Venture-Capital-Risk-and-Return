import numpy as np
import pandas as pd

from scripts.market_api import get_yfinance_time_series, get_fred_time_series
from scripts.utils import *


def load_index_data(ticker: str = None, start=None, end=None, freq: str = None, test=False, prefix=''):
    if test:
        idx = pd.read_csv(prefix+'data/cochrane_data/sptr.csv')['data'][::3]
    else:
        if (start is None) and (end is None):
            data = pd.read_csv(prefix+'data/cochrane_data/sptr.csv')
            idx = data['data']
        else:
            assert ticker is not None, "Ticker must be specified when fetching from API"
            start = (pd.to_datetime(start) - pd.DateOffset(months=2)).strftime('%Y-%m-%d')
            idx = get_yfinance_time_series(ticker, start, end, freq)

    logspret = np.log(idx.pct_change().dropna() + 1)
    return logspret


def load_tbills_data(series_id: str = None, start=None, end=None, freq: str = None):
    if (start is None) and (end is None):
        data = pd.read_csv('./data/cochrane_data/tb3ms.csv')
        rf = data['rf']
    else:
        assert series_id is not None, "Series id must be specified when fetching from API"
        start = (pd.to_datetime(start) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        end = (pd.to_datetime(end) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        rf = get_fred_time_series(series_id, start, end, freq)

    logrf = np.log(1 + rf[::3] / 400)[:-1]
    return logrf


def load_venture_data(roundret: int = 0, round_code: int = 0, industry_code=None, filepath=None, test=False, prefix='./data/', to_date=None, from_date=None):
    assert test or filepath
    if not test:
        data = pd.read_csv(prefix+filepath, parse_dates=['round_date', 'exit_date'])
        
        if to_date is not None:
            before = (data.round_date <= pd.to_datetime(to_date)) & ((data.exit_date <= pd.to_datetime(to_date)) | data.exit_date.isna())
            data = data[before].reset_index(drop=True)
        if from_date is not None:
            after = (data.round_date >= pd.to_datetime(from_date)) & ((data.exit_date >= pd.to_datetime(from_date)) | data.exit_date.isna())
            data = data[after].reset_index(drop=True)
        end_year = max(data.round_date).year
        data.exit_date = pd.to_datetime(data.exit_date, errors="coerce")
        data.round_date = pd.to_datetime(data.round_date, errors="coerce")
    else:
        data = pd.read_csv(prefix+'/data/cochrane_data/returns.csv')
        end_year = 2000

    if (roundret == 1) or (round_code > 0):
        num_rounds = data.groupby("company_num").size()
        round_idx = np.concatenate(num_rounds.apply(lambda x: np.arange(x)).ravel())
        data["round_num"] = round_idx + 1

        if round_code > 0:
            data = data[data["round_num"] == round_code]
            assert not data.empty, f"No data points for rounds = {round_code}"

    if not test and industry_code is not None:
        data = data[data.group_num == industry_code]
        assert not data.empty, f"No data points for industry = {industry_code}"

    # set return to 0 for out of business projects
    data["return_usd"] *= (data["exit_type"] != 3)
    # keep only projects with know exit type
    data = data[data["exit_type"] != -99]
    # keep only projects with correct round dates
    data = data[data["round_date"] != -99]
    # discard round dates after first quarter of end year (need lag for returns)
    data = data[to_decimal_date(data["round_date"]) < end_year + 0.25]

    # set missing exit date when value is before existing round date
    wrong_date_idx = (data["exit_date"] != -99) & (
            to_decimal_date(data["exit_date"]) <= to_decimal_date(data["round_date"]))
    data.loc[wrong_date_idx, "exit_date"] = -99
    # set state to out of business when last return is 0
    wrong_ret_idx = (data["return_usd"] == 0) & (data["exit_type"] != 3)
    data.loc[wrong_ret_idx, "exit_type"] = 3

    data = data.fillna(-99)
    # discard super-extreme returns: hand-checked incorrect values
    # data = data[(data["return_usd"] <= 300)]

    return data.reset_index(drop=True)
