#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Create RMSE time-series for all ensembles
~~~~~~~
"""

from os.path import join
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

model = 'f.e11.FAMIPC5CN'
file_name = f'/mnt/data/CESM1/{model}/input/{{}}.nc'

means = xr.open_dataset(file_name.format('MEANS'))
vars = xr.open_dataset(file_name.format('VARS'))
model_name = means.coords['model'].values.tolist()

for v in means.keys():
    dif = means[v] - vars[v]
    rmse = np.sqrt((dif**2).mean(dim=['lat', 'lon']))

    plt_rmse = rmse.plot(x='time', row='model')
    for ax in plt_rmse.axes.flat:
        ax.set_ylabel(ax.title.get_text().split(' ')[2])
        ax.set_title('')
    plt.tight_layout()
    fig = plt_rmse.fig
    fig.set_size_inches(30, 30)
    fig.savefig(f'figures/mean_real_ts_rmse_{v}.pdf', bbox_inches='tight')
    plt.close(fig)