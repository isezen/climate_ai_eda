#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Plot ensemble time series
~~~~~~~
"""

from matplotlib import pyplot as plt
from os.path import join
import xarray as xr

def plot_ens(x):
    if x.name.startswith('PS'):
        x = x / 100
    m = x.mean(dim=['lat', 'lon'])
    p = m.plot(row='model', x='time')
    p.fig.set_size_inches(30, 30)
    p.fig.savefig(f'figures/plot_ens_{x.name}.pdf', bbox_inches='tight')
    plt.close(p.fig)


model = 'f.e11.FAMIPC5CN'
dt = xr.open_dataset(f'/mnt/data/CESM1{model}/input/VARS.nc')
for _, d in dt.data_vars.items():
    plot_ens(d)
