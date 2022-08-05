import cProfile
from ensurepip import bootstrap
import pstats
import pandas as pd
import pickle
from scripts.preprocess import *
from scripts.minimize_lkh import Model
from scripts.utils import display_return_stats, find_case, transform_params
from scripts.sim import sim

def get_dates(x, test=False):
    start_date, end_date = min(x.round_date), max(x.round_date)
    if test:
        s, e = str(start_date), str(end_date)
        start_date = pd.to_datetime(f"{s[:4]}-{s[4:6]}-{s[-2:]}")
        end_date = pd.to_datetime(f"{e[:4]}-{e[4:6]}-{e[-2:]}")
    return start_date, end_date


def start_opti(x, xc, params, impose_alpha=False, stockidx=1, nopi=0, use_k=1, test=False, maxiter=30, index='^SP500TR'):
    start_date, end_date = get_dates(x, test)
    size = (end_date.to_period(freq='Q') - start_date.to_period(freq='Q')).n + 2
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    if test:
        start = '1987-01-01'
        end = '2000-08-02'

    logrf = load_tbills_data('TB3MS', start, end)
    logmk = load_index_data(index, start, end, '1mo', test)

    c, d = display_return_stats(x)

    minage = 0.25
    logv = np.arange(-7, 7.1, 0.1)
    params[-1] = 0 if nopi == 1 else params[-1]
    mask = [impose_alpha != 1, stockidx is not None, True, use_k != 0, True, True, nopi != 1]
    tpar0 = transform_params(*params, mask)

    model = Model(x, xc, logrf, logmk, minage, c, d, logv, mask, stockidx, use_k, start_year=to_decimal_date(start_date), sample_size=size)
    return model.optimize_likelyhood(tpar0, mask, maxiter=maxiter)


def save_results(results, from_date, to_date, industry, test, bootstrap, pred, index):
    filename = 'src/results/res_'
    if not pred:
        filename += '_' + 'no_pred'
    if from_date is not None:
        filename += 'from_' + from_date[:4]
    if to_date is not None:
        filename += 'to_' + to_date[:4]
    if industry is not None:
        filename += '_' + industry.lower()
    if test:
        filename += '_' + 'cochrane'
    if bootstrap:
        filename += '_' + 'bootstrap'
    if index:
        filename += '_' + index
    
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(results, file)
        
        
def bootstrap(x, xc, params0, n_bstrap, test, index):
    res = {}
    for i in tqdm(range(n_bstrap)):
        x_i = x.sample(frac=0.90, replace=False)
        start_date, end_date = get_dates(x_i, False)
        bootstrap_res = start_opti(x_i.reset_index(drop=True), xc[x_i.index], params0, test=test, index=index)
        res[i] = {'start': start_date, 'end': end_date, 'res': bootstrap_res}
    return res


def main(filepath, params0=None, from_date=None, to_date=None, industries=None, test=False, bootstrap=False, 
         pred=True, use_k=True, use_bkp=True, n_bstrap=100, maxiter=30, index='^SP500TR'):
    
    if params0 is None:
        gamma0 = 0.01
        delta0 = 1
        sigma0 = 0.9
        k0 = 0.1
        a0 = 1
        b0 = 8
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
                res = bootstrap(x_i.reset_index(drop=True), xc[x_i.index], params0, n_bstrap, test, index)
            else:
                res = start_opti(x_i.reset_index(drop=True), xc[x_i.index], params0, test=test, maxiter=maxiter, index=index)
                res = {'start': start_date, 'end': end_date, 'res': res}
            
            save_results(res, from_date, to_date, industry, test, bootstrap, pred, index)
        exit()
    else:
        if bootstrap:
            res = bootstrap(x, xc, params0, n_bstrap, test)
        else:
            start_date, end_date = get_dates(x, False)
            print(f"Industry{'None':>52}")
            print(f"Start date{start_date.strftime('%Y-%m-%d'):>50}")
            print(f"End date{end_date.strftime('%Y-%m-%d'):>52}")
            res = start_opti(x, xc, params0, test=test, maxiter=maxiter, index=index)
            res = {'start': start_date, 'end': end_date, 'res': res}
        
    save_results(res, from_date, to_date, None, test, bootstrap, pred, index)


if __name__ == "__main__":
    index = '^SP500TR'
    main('data.csv', from_date='2010-01-01', index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], index=index)
    main('data.csv', from_date='2010-01-01', bootstrap=True, index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], bootstrap=True, n_bstrap=10, index=index)
    
    index = '^IXIC'
    main('data.csv', from_date='2010-01-01', index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], index=index)
    main('data.csv', from_date='2010-01-01', bootstrap=True, index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], bootstrap=True, n_bstrap=10, index=index)
    
    index = '^RUT'
    main('data.csv', from_date='2010-01-01', index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], index=index)
    main('data.csv', from_date='2010-01-01', bootstrap=True, index=index)
    main('data.csv', from_date='2010-01-01', industries=['Tech', 'Retail', 'Health', 'Other'], bootstrap=True, n_bstrap=10, index=index)
        

###### TODO #######
# run simulations for all datasets: predicitons, cochrane, no pred, bootstrap, 
# same but run by sector
# add missing exit values (go with 1st day market cap, fk it)
# improve selection function
# draw graphs of empiracal CDFs