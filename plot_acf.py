#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles RMSE
~~~~~~~
"""

from os.path import join
import statsmodels.api as sm
from matplotlib import pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
import pandas as pd

model = 'f.e11.FAMIPC5CN'
file_name = f'/mnt/data/CESM1/{model}/input/{{}}.nc'

means = xr.open_dataset(file_name.format('MEANS'))
vars = xr.open_dataset(file_name.format('VARS'))
model_name = means.coords['model'].values.tolist()

for v in means.keys():
    dif = means[v] - vars[v]
    rmse = np.sqrt((dif**2).mean(dim=['lat', 'lon']))

    lags = 24
    df_rmse = rmse.to_dataframe().reset_index('model')
    fg = sns.FacetGrid(data=df_rmse, col='model', col_wrap=2)
    ticks = list(range(0, lags + 1, 6))
    for ax, m in zip(fg.axes, model_name):
        df2 = df_rmse[df_rmse['model'] == m]
        sm.graphics.tsa.plot_acf(df2.iloc[:, 1].values.squeeze(), ax=ax, lags=lags,
            title=m)
        ax.grid(color='lightgray', linestyle='--', linewidth=1)
        plt.xticks(ticks, ticks)

    plt.tight_layout()
    fig = fg.figure
    fig.set_size_inches(5, 10)
    fig.savefig(f'figures/mean_real_acf_rmse_{v}.pdf', bbox_inches='tight')
    plt.close(fig)
