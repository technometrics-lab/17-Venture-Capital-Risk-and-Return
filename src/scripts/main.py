import pandas as pd
import numpy as np
from scripts.preprocess import *
from scripts.minimize_lkh import Model
from scripts.utils import display_return_stats, find_case, transform_params, save_results

def get_dates(x, test=False):
    """Returns the start and end date for the given dataset.

    Args:
        x (DataFrame): dataset of observations
        test (bool, optional): Whether the dataset is Cochrane test set or not. Defaults to False.

    Returns:
        (str, str): begining and end dates, format yy-mm-dd
    """    
    start_date, end_date = min(x.round_date), max(x.round_date)
    if test:
        s, e = str(start_date), str(end_date)
        start_date = pd.to_datetime(f"{s[:4]}-{s[4:6]}-{s[-2:]}")
        end_date = pd.to_datetime(f"{e[:4]}-{e[4:6]}-{e[-2:]}")
    return start_date, end_date


def start_opti(x, xc, params, impose_alpha=False, stockidx=1, nopi=0, use_k=1, test=False, maxiter=30, benchmark='^SP500TR'):
    """Starts the MLE procedure, given an initial dataset. Automatically infers dates, downloads market data, and pre-process VC data.
    Expected parameters are: gamma, delta, sigma, k, a, b, pi, used as initial values for the optimization.
    Data for risk free rate is the 3 month t-bill from FRED (quarterly)
    Data for public equity indices are fetched from yahoo finance.
    The main dataset columns must be: company_id, group, segment, round_date, raised_usd, postvalue_usd, exit_date, exit_type, exit_value, return_usd

    Args:
        x (DataFrame): dataset of observations
        xc (DataFrame): categories for each observations (between 1 and 6)
        params (list): list of parameters for the `find_likelihood` function
        impose_alpha (bool, optional): Test for alpha=0. Defaults to False.
        stockidx (int, optional): Use market as benchmark. Defaults to 1.
        nopi (int, optional): Use error detection (pi). Defaults to 0.
        use_k (int, optional): Account for failures. Defaults to 1.
        test (bool, optional): Test set or true dataset. Defaults to False.
        maxiter (int, optional): Maximum number of iterations for the minimization function. Defaults to 30.
        benchmark (str, optional): Which market benchmark to use. Defaults to `^SP500TR`.

    Returns:
        DataFrame: Result dataframe containing parameter values and standard errors
    """    
    start_date, end_date = get_dates(x, test)
    size = (end_date.to_period(freq='Q') - start_date.to_period(freq='Q')).n + 2
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    if test:
        start = '1987-01-01'
        end = '2000-08-02'

    logrf = load_tbills_data('TB3MS', start, end)
    logmk = load_index_data(benchmark, start, end, '1mo', test)

    c, d = display_return_stats(x)

    minage = 0.25 # minimum round age since beginning of observations (in quarter). Minimum is 0.25 (1 quarter)
    logv = np.arange(-7, 7.1, 0.1)
    params[-1] = 0 if nopi == 1 else params[-1]
    mask = [impose_alpha != 1, stockidx is not None, True, use_k != 0, True, True, nopi != 1]
    tpar0 = transform_params(*params, mask)

    model = Model(x, xc, logrf, logmk, minage, c, d, logv, mask, stockidx, use_k, start_year=to_decimal_date(start_date), sample_size=size)
    return model.optimize_likelyhood(tpar0, mask, maxiter=maxiter)
        
        
def run_bootstrap(x, xc, params0, n_bstrap, test, benchmark):
    """Runs bootstrap iterations

    Args:
        x (DataFrame): dataset of observations
        xc (DataFrame): categories for each observations (between 1 and 6)
        params (list): list of parameters for the find_likelihood function
        n_bstrap (int): number of bootstrap iterations to perform
        test (boolean): use Cochrane test dataset or not
        benchmark (str): Which market benchmark to use. Defaults to '^SP500TR'.

    Returns:
        dict: dictionnary containing all bootstrap results, not aggregated
    """    
    frac = 0.8
    res = {}
    for i in range(n_bstrap):
        idx = np.sort(np.random.choice(x.index, replace=False, size=int(x.shape[0] * frac)))
        x_i = x.iloc[idx,:]
        start_date, end_date = get_dates(x_i, False)
        print(f'Bootsrap iteration{i+1:>39}/{n_bstrap}')
        print(f"Start date{start_date.strftime('%Y-%m-%d'):>50}")
        print(f"End date{end_date.strftime('%Y-%m-%d'):>52}")
        
        xc_i = xc[x_i.index].copy()
        x_i = x_i.reset_index(drop=True).copy()
        
        bootstrap_res = start_opti(x_i, xc_i, params0, test=test, maxiter=30, benchmark=benchmark)
        res[i] = {'start': start_date, 'end': end_date, 'res': bootstrap_res}
    return res


def main(filepath, params0=None, from_date=None, to_date=None, industries=None, test=False, bootstrap=False, 
         pred=True, use_k=True, use_bkp=True, n_bstrap=100, maxiter=30, benchmark='^SP500TR'):
    
    if params0 is None:
        gamma0 = 0.01
        delta0 = 1
        sigma0 = 0.9
        k0 = 0.1
        a0 = 1
        b0 = 10
        pi0 = 0.1
        
        params0 = [gamma0, delta0, sigma0, k0, a0, b0, pi0]
    else:
        assert len(params0) == 7
    
    print(f"{' INFO ':=^60}")
    print(f"Dataset info{filepath:>48}")
    x = load_venture_data(filepath=filepath, from_date=from_date, to_date=to_date, test=test)
    x["ddate"] = to_decimal_date(x["round_date"])
    x = x.sort_values(by=["ddate", "company_num"]).drop(columns=["ddate"]).reset_index(drop=True)
    xc = find_case(x, use_k, use_bkp)
    
    if industries:
        if not isinstance(industries, list):
            industries = [industries]
            
        for industry in industries:
            x_i = x[x.group_num == industry]
            print(f"Industry{industry:>52}")
            start_date, end_date = get_dates(x_i, False)
            print(f"Start date{start_date.strftime('%Y-%m-%d'):>50}")
            print(f"End date{end_date.strftime('%Y-%m-%d'):>52}")
            
            if bootstrap:
                res = run_bootstrap(x_i.reset_index(drop=True), xc[x_i.index], params0, n_bstrap, test, benchmark)
            else:
                res = start_opti(x_i.reset_index(drop=True), xc[x_i.index], params0, test=test, maxiter=maxiter, benchmark=benchmark)
                res = {'start': start_date, 'end': end_date, 'res': res}
            
            save_results(res, from_date, to_date, industry, test, bootstrap, pred, benchmark)
        exit()
    else:
        if bootstrap:
            res = run_bootstrap(x, xc, params0, n_bstrap, test, benchmark)
        else:
            start_date, end_date = get_dates(x, False)
            print(f"Industry{'None':>52}")
            print(f"Start date{start_date.strftime('%Y-%m-%d'):>50}")
            print(f"End date{end_date.strftime('%Y-%m-%d'):>52}")
            res = start_opti(x, xc, params0, test=test, maxiter=maxiter, benchmark=benchmark)
            res = {'start': start_date, 'end': end_date, 'res': res}
        
    save_results(res, from_date, to_date, None, test, bootstrap, pred, benchmark)


if __name__ == "__main__":
    benchmark = '^SP500TR'
    main('data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)

    benchmark = '^IXIC'
    main('data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)

    benchmark = '^RUT'
    main('data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], benchmark=benchmark)
    main('data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)
    
    benchmark = '^SP500TR'
    main('security_data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('security_data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)
    benchmark = '^IXIC'
    main('security_data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('security_data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)
    benchmark = '^RUT'
    main('security_data.csv', from_date='2010-01-01', benchmark=benchmark)
    main('security_data.csv', from_date='2010-01-01', bootstrap=True, benchmark=benchmark)
   
