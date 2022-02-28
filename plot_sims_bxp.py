#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Create plots for AI project NC files
~~~~~~~
Python script to create plots
"""

import os
from os.path import join
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import pandas as pd

metric_names = ["ME", "MAE", "RMSE", "MAXE", "PAE"]
pred_names = ['Mean', 'AI']

nsim = 6
val_indices = slice(1306, 1406)
model = 'f.e11.FAMIPC5CN'
data_path = f'/mnt/data/CESM1/{model}'

out_path = join(data_path, 'output')
in_path = join(data_path, 'input')

sim_names = [f'sim{i}' for i in range(1, nsim + 1)]
sim_dim = pd.Index(sim_names, name='sim')
sim_paths = [os.path.join(out_path, sn, 'pred.nc') for sn in sim_names]
sims = xr.concat([xr.open_dataset(sp)['TS'] for sp in sim_paths], sim_dim)

# validation dataset
means = xr.open_dataset(join(in_path, 'MEANS.nc'))['TS']
means = means[:,val_indices].assign_coords(sim='Mean').expand_dims(sim=1)
sims = xr.concat([means, sims], dim='sim')

# real dataset
reals = xr.open_dataset(join(in_path, 'VARS.nc'))['TS']
reals = reals[:,val_indices]

# Prediction vs real
x = xr.concat([s - reals for s in sims], dim='sim')
xa = abs(x)
del sims, means, reals

mae = xa.mean(dim=['time', 'lat', 'lon'])
maxe = xa.max(dim=['time', 'lat', 'lon'])
mse = (x**2).mean(dim=['time', 'lat', 'lon'])
rmse = np.sqrt(mse)

metrics = xr.concat([mae, mse, rmse, maxe],
                    pd.Index(['mae', 'mse', 'rmse', 'maxe'], 
                             name='metric'))
df = metrics.to_dataframe().reset_index()
fg = sns.catplot(data=df, x='sim', y='TS',
                 col='metric', col_wrap=2, kind='bar', sharey=False)
for mi, ma, ax in zip([0.76, 2.6, 1.6, 19],
                      [0.91, 3.0, 1.74, 23], fg.axes):
    ax.set(ylim=(mi, ma))
fig = fg.figure
fig.tight_layout(pad=2.0)
fig.set_size_inches(9, 7)
fig.savefig(f'figures/metrics.pdf', bbox_inches='tight')
plt.close(fig)

metrics = {'mae': mae, 'mse': mse, 'rmse': rmse}
for k, v in metrics.items():
    df = v.to_dataframe().reset_index()
    min_max = v.min(axis=0).min().values.tolist()
    min_lim = (min_max - 0.1) * 100 // 10 / 10
    fg = sns.catplot(data=df, x='sim', y='TS',
                     col='model', col_wrap=2, kind='bar')
    fg.set(ylim=(min_lim, None))
    fig = fg.figure
    fig.tight_layout(pad=2.0)
    fig.set_size_inches(10, 10)
    fig.savefig(f'figures/{k}_mean_ai1.pdf', bbox_inches='tight')
    plt.close(fig)

    fg = sns.catplot(data=df, x='model', y='TS',
                     col='sim', col_wrap=2, kind='bar')
    fg.set(ylim=(min_lim, None))
    fig = fg.figure
    fig.tight_layout(pad=2.0)
    fig.set_size_inches(13, 10)
    fig.savefig(f'figures/{k}_mean_ai2.pdf', bbox_inches='tight')
    plt.close(fig)

# See: https://matplotlib.org/3.1.0/gallery/statistics/bxp.html
# See: https://stackoverflow.com/questions/29895754/buildling-boxplots-incrementally-from-large-datasets
for k, v in {'dif': x, 'abs': xa}.items():
    rng = np.linspace(start=0, stop=1, num=10000)
    v2 = v.quantile(rng, dim=['time', 'lat', 'lon'])
    df = v2.to_dataframe().reset_index()
    # df = v.to_dataframe().reset_index()
    fg = sns.catplot(data=df, x='sim', y='TS', col='model', col_wrap=2,
                 kind='box', sharey=False, showfliers=False)
    fig = fg.figure
    fig.tight_layout(pad=1.0)
    fig.set_size_inches(10, 11)
    fig.savefig(f'figures/bxp_sims_{k}1.pdf', bbox_inches='tight')
    plt.close(fig)

    fg = sns.catplot(data=df, x='model', y='TS', col='sim', col_wrap=2,
                     kind='box', sharey=False, showfliers=False)
    fig = fg.figure
    fig.tight_layout(pad=1.0)
    fig.set_size_inches(13, 10)
    fig.savefig(f'figures/bxp_sims_{k}2.pdf', bbox_inches='tight')
    plt.close(fig)

