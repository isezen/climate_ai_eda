#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
regrid netcdf data
~~~~~~~
Python script to regrid netcdf data
"""

from os.path import join
# from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import xesmf as xe


data_path = '/mnt/data/'
latlon_file = join(data_path, 'CESM1/f.e11.FAMIPC5CN/input', 'MEANS.nc')
target_file = join(data_path, 'ERA5', 'ERA5_mar_T2_1979_2021.nc')
out_file = join(data_path, 'ERA5', 'ERA5_mar_T2_1979_2021_regrid.nc')

ncf1 = xr.open_dataset(latlon_file, cache=False)
lats, lons = ncf1.lat.values, ncf1.lon.values

ncf2 = xr.open_dataset(target_file, cache=False)

nc_out = xr.Dataset(
    {
        "latitude": (["latitude"], lats),
        "longitude": (["longitude"], lons),
    }
)

regridder = xe.Regridder(ncf2, nc_out, "bilinear")
nc_out = regridder(ncf2)

encoding = {'dtype': np.dtype('float32'), 'zlib': True, 'complevel': 5}
nc_out.to_netcdf(out_file, encoding={'t2m': encoding})
