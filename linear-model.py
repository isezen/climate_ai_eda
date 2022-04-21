#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Linear Regression
~~~~~~~
Python script to make Linear Regression on models
"""

import numpy as np
import pandas as pd
import xarray as xr
from os.path import join
from os import makedirs as mkdir
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


encoding = {'dtype': np.dtype('float32'), 'zlib': True, 'complevel': 5}
val_indices = slice(1306, 1406)
model = 'f.e11.FAMIPC5CN'
data_path = f'/mnt/data/CESM1/{model}'

out_path = join(data_path, 'output')
in_path = join(data_path, 'input')
sim_path = join(out_path, 'sim0')

data = xr.open_dataset(join(in_path, 'VARS.nc'))['TS']
data2 = data[:, val_indices].copy()
data2.data = np.zeros(data2.shape)

for i, ts in enumerate(data):
    print(i)
    for la in range(0, 192):
        for lo in range(0, 288):
            v = ts[:, la, lo]
            dates = pd.to_datetime(v.coords['time'])
            years = dates.year.values.tolist()
            months = dates.month.values.tolist()
            v = v.assign_coords({'year': ('time', years),
                                 'month': ('time', months)})
            v2 = v[val_indices]

            df = v.to_dataframe().reset_index()
            df2 = v2.to_dataframe().reset_index()

            for j in range(1, 13):
                # indices = (df['month'] == j).values
                df3 = df.iloc[(df['month'] == j).values]
                cond = df2['month'] == j
                indices = df2.index[cond]
                df4 = df2.iloc[indices]

                X = df3[['year']]
                y = df3['TS']
                P = df4[['year']]
                lm = LinearRegression()
                lm.fit(X, y)
                pred = np.array(lm.predict(P))
                df2.loc[indices, 'TS'] = pred
            data2[i, :, la, lo] = df2['TS'].values

mkdir(sim_path, exist_ok=True)
data2.to_netcdf(join(sim_path, 'pred.nc'), encoding={'TS': encoding})

