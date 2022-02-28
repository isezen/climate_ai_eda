#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Create plots for AI project NC files
~~~~~~~
Python script to create plots
"""

from os.path import join
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import pandas as pd

def bxp_stats(x):
    y = x.quantile([0.25, 0.5, 0.75], dim=['lat', 'lon'])
    y = y.rename({'quantile': 'stat'})
    y.coords['stat'] = ['q25', 'median', 'q75']
    iqr15 = 1.5 * abs(y[2] - y[1])
    lw = y[0] - iqr15
    uw = y[2] + iqr15
    mean = x.mean(dim=['lat', 'lon'])
    mean = mean.assign_coords(stat='mean')
    lw.coords['stat'] = 'lw'
    uw.coords['stat'] = 'uw'
    return xr.concat([mean, lw, y, uw], dim = 'stat')

model = 'f.e11.FAMIPC5CN'
file_name = f'/mnt/data/CESM1/{model}/input/{{}}.nc'

means = xr.open_dataset(file_name.format('MEANS'))
vars = xr.open_dataset(file_name.format('VARS'))
model_name = means.coords['model'].values.tolist()

for v in means.keys():
    dif = means[v] - vars[v]
    a = abs(dif)
    stats = xr.concat([bxp_stats(m.drop('model')) for m in dif], dim='model')
    stats = stats.squeeze(drop=True)
    stats = stats.assign_coords(model=model_name)

    q = [stats[:,:,i].to_dataframe().reset_index() for i in [1, 2, 4, 5]]
    lines_down = pd.concat([q[0].iloc[:,[0,1,3]], q[1].iloc[:,3]], axis=1)
    lines_up = pd.concat([q[2].iloc[:,[0,1,3]], q[3].iloc[:,3]], axis=1)

    df = stats.to_dataframe().reset_index()

    df2 = df[df['stat'] == 'mean']
    df2 = df2.rename(columns={df2.columns.values[3]: 'RMSE'})

    # fg = sns.FacetGrid(data=df2, hue="stat", row='model')
    fg = sns.relplot(data=df2, x="time", y="RMSE", hue="stat",
                     s=1, row='model', kind="scatter")

    fg.set(ylim=(stats.min(), stats.max()))

    for i in range(len(model_name)):
        ax = fg.axes.flat[i]
        ax.set_ylabel(ax.title.get_text().split(' ')[2])
        ax.set_title('')
        m = model_name[i]
        lines = lines_up.loc[lines_up['model'] == m]
        y1 = lines.iloc[:,2].tolist()
        y2 = lines.iloc[:,3].tolist()
        x = lines.iloc[:,1].tolist()
        for j in range(len(x)):
            if abs(y1[j] - y2[j]) > 0:
                ax.plot([x[j], x[j]], [y1[j], y2[j]], '-', linewidth=0.7)

        lines = lines_down.loc[lines_down['model'] == m]
        y1 = lines.iloc[:,2].tolist()
        y2 = lines.iloc[:,3].tolist()
        x = lines.iloc[:,1].tolist()
        for j in range(len(x)):
            if abs(y1[j] - y2[j]) > 0:
                ax.plot([x[j], x[j]], [y1[j], y2[j]], '-', linewidth=0.7)

    plt.tight_layout()
    fig = fg.figure
    fig.set_size_inches(30, 30)
    fig.savefig(f'figures/mean_real_time_bxp_{v}.pdf', bbox_inches='tight')
    plt.close(fig)
