import pandas as pd
import numpy as np
import pystan


def data4GaussianMixture(file_path, K=10):
    df = pd.read_csv(file_path, index_col=0).T
    return {
        'N': df.shape[0],
        'K': K,
        'D': df.shape[1],
        'y': df.values,
        'alpha0': 0.1,
        'mu_sigma0': 10,
        'sigma_sigma0': 5,
    }


def data4Student_tMixture(file_path, K):
    d = data4GaussianMixture(file_path, K)
    d.pop('sigma_sigma0')
    d['sigma_shape'] = 5
    d['sigma_scale'] = 5
    d['nu'] = 3
    return d


def stan2DF(fit, num_cluster=10):
    '''
    transforms the output of Stan's variational inference method to
    a pandas DataFrame
    '''
    df = pd.DataFrame({}, columns=range(10))
    for name, val in zip(fit['mean_par_names'], fit['mean_pars']):
        if 'z' in name:
            row, col = par_name2int(name)
            df.loc[row, col - 1] = np.exp(val)
    df = df.apply(lambda x: x/x.sum(), axis=1)
    return df.applymap(lambda x: round(x, 3))


def par_name2int(name):
    return [int(n) for n in name.split('[')[1].split(']')[0].split(',')]


def elbo_df(sm, data_path, cl_range, num_of_epochs):
    elbo = pd.DataFrame({}, columns=cl_range, index=num_of_epochs)
    for cluster_num in elbo.columns:
        for epoch in elbo.index:
            d = data4Student_tMixture(data_path, cluster_num)
            sm.vb(d, sample_file='dummy.csv',
                  diagnostic_file=f'results/ppca/diagnostic_{cluster_num}.csv')
            elbo_val = elbo_from_diagnostic(
                f'results/ppca/diagnostic_{cluster_num}.csv')
            elbo.loc[epoch, cluster_num] = elbo_val
    return elbo


def elbo_from_diagnostic(file_path):
    val = pd.read_csv(file_path, skiprows=23)
    return float(val.columns[2])
