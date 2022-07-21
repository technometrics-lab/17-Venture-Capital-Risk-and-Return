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


def main(x, xc, params, impose_alpha=False, stockidx=1, nopi=0, use_k=1, test=False, maxiter=30):
    start_date, end_date = get_dates(x, test)
    size = (end_date.to_period(freq='Q') - start_date.to_period(freq='Q')).n + 2
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    if test:
        start = '1987-01-01'
        end = '2000-08-02'

    logrf = load_tbills_data('TB3MS', start, end)
    logmk = load_index_data('^SP500TR', start, end, '1mo', test)

    c, d = display_return_stats(x)

    print(f'Percent of valuations (ipo, acquired, new round) that have good data: {d * 100:.2f}%\n\n')

    minage = 0.25
    logv = np.arange(-7, 7.1, 0.1)
    params[-1] = 0 if nopi == 1 else params[-1]
    mask = [impose_alpha != 1, stockidx is not None, True, use_k != 0, True, True, nopi != 1]
    tpar0 = transform_params(*params, mask)

    model = Model(x, xc, logrf, logmk, minage, c, d, logv, mask, stockidx, use_k, start_year=to_decimal_date(start_date), sample_size=size)
    return model.optimize_likelyhood(tpar0, mask, maxiter=maxiter)


if __name__ == "__main__":
    N = 50
    gamma0 = 0.01
    delta0 = 1.5
    sigma0 = 0.9
    k0 = 0.1
    a0 = 1
    b0 = 3
    pi0 = 0.01
    
    use_k, bankhand, industry = 1, 2, None #['Tech', 'Retail', 'Health', 'Other']
    pred, bootstrap, test = True, False, False
    
    params0 = [gamma0, delta0, sigma0, k0, a0, b0, pi0]
    
    x = load_venture_data(pred=pred, test=test)
    x["ddate"] = to_decimal_date(x["round_date"])
    x = x.sort_values(by=["ddate", "company_num"]).drop(columns=["ddate"]).reset_index(drop=True)
    xc = find_case(x, use_k, bankhand)
    
    if bootstrap:
        res = {}
        start_date, end_date = get_dates(x, False)
        for i in tqdm(range(N)):
            x_i = x.sample(frac=0.90, replace=False)
            bootstrap_res = main(x_i.reset_index(drop=True), xc[x_i.index], params0, test=False)
            res[i] = bootstrap_res
    elif industry:
        if not isinstance(industry, list):
            industry = [industry]
            
        for ind in industry:
            x_i = x[x.group_num == ind]
            print(f'Running model for industry: {ind}')
            start_date, end_date = get_dates(x_i, False)
            res = main(x_i.reset_index(drop=True), xc[x_i.index], params0, test=test)
            res = {'start': start_date, 'end': end_date, 'res': res}
            
            filename = 'res_' + ind
            with open(filename + '.pkl', 'wb') as file:
                pickle.dump(res, file)
        exit()
    else:
        start_date, end_date = get_dates(x, test)
        res = main(x, xc, params0, test=test)

        
    res = {'start': start_date, 'end': end_date, 'res': res}
    filename = 'res'
    if test:
        filename += '_cochrane'
    elif bootstrap:
        filename += '_bootstrap'
    elif not pred:
        filename += '_nopred'
        
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(res, file)
        
        

###### TODO #######
# run simulations for all datasets: predicitons, cochrane, no pred, bootstrap, 
# same but run by sector
# add missing exit values (go with 1st day market cap, fk it)
# improve selection function
# draw graphs of empiracal CDFs