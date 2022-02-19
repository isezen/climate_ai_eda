#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles
~~~~~~~
Python script to create plots
"""

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
vars = {v: None for v in variables}
for v in variables:
    fn = join(path_from, v, f'{model}.nc')
    fn = fn.format('{}', f'.{v}')
    data = [xr.open_dataset(fn.format(f'.{m}'))[v] for m in model_name]
    data = xr.concat(data, dim='model')
    data = data.assign_coords(model=model_name)
    attrs = data.attrs
    if v == 'TS':
        data = data  - 272.15
        attrs['units'] = 'C'
    data.coords['time'] = data.indexes['time'].to_datetimeindex()
    data.attrs = attrs
    vars[v] = data

# TOTAL PREC
prect = vars['PRECC']+ vars['PRECL']
prect.name = 'PRECT'
vars['PRECT'] = prect

# VARIABLES
variables = ['TS', 'PRECC', 'PRECL', 'PRECT', 'PS', 'PSL', 'UBOT', 'VBOT']
enc = {v: encoding for v in variables}

vars = xr.merge(list(vars.values()))
vars.to_netcdf(join(path_to, "VARS.nc"), encoding=enc)
del data, prect

# Means
my_means = {v: None for v in variables}
for v in variables:
    data = vars[v]
    means = [data[indices != i].mean(axis=0) for i in indices]
    means = xr.concat(means, dim='model')
    means = means.assign_coords(model=model_name)
    means.name = v
    my_means[v] = means

means = xr.merge(list(my_means.values()))
means.to_netcdf(join(path_to, "MEANS.nc"), encoding=enc)
del data, my_means

# Differences
dif = means['TS'] - vars['TS']
del means, vars
dif.name = 'TS_dif'
dif.to_netcdf(join(path_to, f"{dif.name}.nc"), 
              encoding={dif.name: encoding})


# ds =data.to_dataset()
# encoding = {}
# encoding_keys = ("_FillValue", "dtype", "scale_factor", "add_offset", "grid_mapping")
# for data_var in ds.data_vars:
#     encoding[data_var] = {key: value for key, value in ds[data_var].encoding.items() if key in encoding_keys}
#     encoding[data_var].update(zlib=True, complevel=5)
