import yfinance
import dotenv
import requests
import json
import pandas as pd


def get_fred_time_series(series_id: str, start: str = None, end: str = None, freq: str = None) -> pd.DataFrame:
    fred_api_key = dotenv.get_key('./.env', "FRED_API_KEY")
    freqs = ['d', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth',
             'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem']
    url = (f'https://api.stlouisfed.org/fred/series/observations?'
           f'series_id={series_id}&api_key={fred_api_key}&file_type=json')

    if start is not None:
        url += f'&observation_start={start}'
    if end is not None:
        url += f'&observation_end={end}'
    if freq is not None:
        assert freq in freqs, f"Invalid frequency, must be one of {freqs}"
        url += f'&frequency={freq}'

    r = requests.get(url)
    assert r.status_code == 200, f"Request error: code [{r.status_code}]"
    data = json.loads(r.text)['observations']

    dates, values = [], []
    for obs in data:
        dates.append(obs['date'])
        values.append(float(obs['value']))
    return pd.DataFrame({'date': pd.to_datetime(dates), series_id: values}).set_index('date')


def get_yfinance_time_series(ticker: str, start: str = None, end: str = None, freq: str = None) -> pd.DataFrame:
    data = yfinance.download(ticker, start=start, end=end, interval=freq, progress=False)['Adj Close']
    data = data.to_frame()
    data.index.name = 'date'
    data.columns = [ticker]
    return data
