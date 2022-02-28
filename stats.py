#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles
~~~~~~~
Python script to create plots
"""

from matplotlib import pyplot as plt
# import statsmodels.api as sm
# from matplotlib import pyplot as plt
from os.path import join
import xarray as xr
# import seaborn as sns
import numpy as np
import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

def summary(x, axis=None):
    if axis is not None:
        dims = list(x.coords.dims)
        dims.pop(axis)
    else:
        dims = None

    model_name = x.coords['model']

    min = x.min(dim=dims)
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95], dim=dims)
    mean = x.mean(dim=dims)
    max = x.max(dim=dims)

    stat_names = ['Min','Q05', 'Q25', 'Median', 'Mean', 'Q75', 'Q95', 'Max']
    if dims is None:
        df = pd.DataFrame(
            np.r_[min.values, q[0:3].values, mean.values,
                  q[3:5].values, max.values],
            index=stat_names,
            columns=[x.name]).T
    else:
        min_df = min.to_dataframe().reset_index().iloc[:, 1]
        max_df = max.to_dataframe().reset_index().iloc[:, 1]
        mean_df = mean.to_dataframe().reset_index().iloc[:, 1]
        q_df = pd.concat([i.to_dataframe().reset_index().iloc[:, 2] for i in q],
                         axis=1)
        df = pd.concat([min_df, q_df.iloc[:,0:3], 
                        mean_df, q_df.iloc[:,3:5], max_df], axis=1)
        df.index = model_name
        df.columns = stat_names
    return df

variable = 'TS'
model = 'f.e11.FAMIPC5CN.f09_f09.historical.toga.cam.h0.188001-200512'
path_from = join('/', 'mnt', 'data', 'CESM1', model)
fn = join(path_from, f'{variable}.nc')
data = xr.open_dataarray(fn)

# Max Difference between ensembles
max_dif = {n.coords['model'].values.tolist(): {m.coords['model'].values.tolist(): 
        abs(m - n).max().values.tolist() for m in data} for n in data}
df_max_dif = pd.DataFrame(max_dif)


# Summary
summary(data)
summary(data, axis=0)

        