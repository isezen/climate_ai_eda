#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles
~~~~~~~
Python script to create plots
"""

from os import remove
from os.path import join
from os import makedirs as mkdir
import xarray as xr
import numpy as np
import pandas as pd

encoding = {'dtype': np.dtype('float32'), 'zlib': True, 'complevel': 5}
indices = np.arange(10)
model_name = np.array([f'ens{i:02d}' for i in (indices + 1)])
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0{{}}.188001-200512'
model_short = '.'.join(model.split('.')[0:3])
base_dir = '/mnt/data/CESM1'
path_to = join(base_dir, model_short, 'input')
path_from = base_dir
mkdir(path_to, exist_ok=True)

variables = ['TS', 'PRECC', 'PRECL', 'PS', 'PSL', 'UBOT', 'VBOT']
for i, v in enumerate(variables):
    fn = join(path_from, v, f'{model}.nc')
    fn = fn.format('{}', f'.{v}')
    data = [xr.open_dataset(fn.format(f'.{m}'), cache=False)[v] 
            for m in model_name]
    data = xr.concat(data, dim='model')
    data = data.assign_coords(model=model_name)
    attrs = data.attrs
    if v == 'TS':
        data = data  - 272.15
        attrs['units'] = 'C'
    if v == 'UBOT' or v == 'VBOT':
        data = data.squeeze().drop('lev')
    # if v.startswith('PS'):
    #     data = data / 100
    #     attrs['units'] = 'hPa'
    data.coords['time'] = data.indexes['time'].to_datetimeindex()
    data.attrs = attrs
    mode = 'w' if i == 0 else 'a'
    data.to_netcdf(join(path_to, 'VARS.nc'),
                   encoding={v: encoding}, mode=mode)
del data

with xr.open_dataset(join(path_to, 'VARS.nc'), cache=False) as ds:
    prect = ds['PRECC'] + ds['PRECL']  # TOTAL PREC
    prect.name = 'PRECT'
    psl_ps = (ds['PSL'] - ds['PS'])/100  # PSL - PS
    psl_ps.name = 'PSL_PS'
prect.to_netcdf(join(path_to, 'VARS.nc'), 
                encoding={'PRECT': encoding}, mode='a')
psl_ps.to_netcdf(join(path_to, 'VARS.nc'),
                 encoding={'PSL_PS': encoding}, mode='a')
del prect, psl_ps

# Create Means
vars = xr.open_dataset(join(path_to, 'VARS.nc'), cache=False)
for i, (k, d) in enumerate(vars.items()):
    d = [d[indices != i].mean(axis=0) for i in indices]
    d = xr.concat(d, dim='model')
    d = d.assign_coords(model=model_name)
    d.name = k
    mode = 'w' if i == 0 else 'a'
    d.to_netcdf(join(path_to, 'MEANS.nc'),
                encoding={k: encoding}, mode=mode)
del d


# Differences
means = xr.open_dataset(join(path_to, 'MEANS.nc'))
dif = means['TS'] - vars['TS']
dif.name = 'dif_TS'
dif.to_netcdf(join(path_to, f'{dif.name}.nc'), 
              encoding={dif.name: encoding})


