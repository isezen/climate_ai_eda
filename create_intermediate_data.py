#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles
~~~~~~~
Python script to create plots
"""

from os.path import join
import xarray as xr
import numpy as np
import pandas as pd

def save_to_nc(x):
    encoding = {'dtype': np.dtype('float32'), 'zlib': True, 'complevel': 5}
    x.to_netcdf(join(path_to, f"{x.name}.nc"),
                encoding={x.name: encoding})

indices = np.arange(10)
model_name = np.array([f'ens{i:02d}' for i in (indices + 1)])
variables = ['TS', 'PRECC', 'PRECL', 'PS', 'PSL', 'UBOT', 'VBOT']
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0{{}}.188001-200512'
path_to = join('data', f"{model.format('', '')}")

for variable in variables:
    path_from = join('/', 'mnt', 'data', 'CESM1', variable)
    fn = join(path_from, f'{model}.nc')
    fn = fn.format('{}', f'.{variable}')
    data = [xr.open_dataset(fn.format(f'.{m}'))[variable] for m in model_name]
    data = xr.concat(data, dim='model')
    data = data.assign_coords(model=model_name)
    attrs = data.attrs
    if variable == 'TS':
        data = data  - 272.15
        attrs['units'] = 'C'
    data.coords['time'] = data.indexes['time'].to_datetimeindex()
    data.attrs = attrs
    save_to_nc(data)
del data

# TOTAL PREC
precc = xr.open_dataarray(join(path_to, 'PRECC.nc'))
precl = xr.open_dataarray(join(path_to, 'PRECL.nc'))
prect = precc + precl
prect.name = 'PRECT'
save_to_nc(prect)
del precc, precl, prect

# Means
data = xr.open_dataarray(join(path_to, 'TS.nc'))
means = [data[indices != i].mean(axis=0) for i in indices]
means = xr.concat(means, dim='model')
means = means.assign_coords(model=model_name)
means.name = 'MEANS_TS'
save_to_nc(means)

# Differences
dif = means - data
dif.name = 'TS_dif'
save_to_nc(dif)


# ds =data.to_dataset()
# encoding = {}
# encoding_keys = ("_FillValue", "dtype", "scale_factor", "add_offset", "grid_mapping")
# for data_var in ds.data_vars:
#     encoding[data_var] = {key: value for key, value in ds[data_var].encoding.items() if key in encoding_keys}
#     encoding[data_var].update(zlib=True, complevel=5)
