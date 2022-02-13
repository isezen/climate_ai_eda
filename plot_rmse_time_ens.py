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

variable = 'TS'
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0.{variable}.188001-200512'
path_to = join('data', f"{model.format('')}")
fn = join(path_to, 'dif.nc')

dif = xr.open_dataset(fn)[variable]
a = abs(dif)
rmse = np.sqrt((a**2).mean(dim=['lat', 'lon']))

plt_rmse = rmse.plot(x='time', row='model')
for ax in plt_rmse.axes.flat:
    ax.set_ylabel(ax.title.get_text().split(' ')[2])
    ax.set_title('')
plt.tight_layout()
fig = plt_rmse.fig
fig.set_size_inches(30, 30)
fig.savefig(f'figures/mean_real_rmse_time.pdf', bbox_inches='tight')
plt.close(fig)
