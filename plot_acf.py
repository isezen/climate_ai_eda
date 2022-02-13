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

variable = 'TS'
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0.{variable}.188001-200512'
path_to = join('data', f"{model.format('')}")
fn = join(path_to, 'dif.nc')

dif = xr.open_dataset(fn)[variable]
model_name = dif.coords['model'].values.tolist()
a = abs(dif)
rmse = np.sqrt((a**2).mean(dim=['lat', 'lon']))

lags = 24
df_rmse = rmse.to_dataframe().reset_index('model')
fg = sns.FacetGrid(data=df_rmse, col='model', col_wrap=3)
ticks = list(range(0, lags + 1, 6))
for ax, m in zip(fg.axes, model_name):
    df2 = df_rmse[df_rmse['model'] == m]
    sm.graphics.tsa.plot_acf(df2['TS'].values.squeeze(), ax=ax, lags=lags,
        title=m)
    ax.grid(color='lightgray', linestyle='--', linewidth=1)
    plt.xticks(ticks, ticks)

plt.tight_layout()
fig = fg.figure
fig.set_size_inches(10, 10)
fig.savefig(f'figures/mean_real_acf_rmse.pdf', bbox_inches='tight')
plt.close(fig)
