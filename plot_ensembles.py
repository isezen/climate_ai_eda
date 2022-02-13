#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
ACF plot for ensembles
~~~~~~~
Python script to create plots
"""

import statsmodels.api as sm
from matplotlib import pyplot as plt
from os.path import join
import xarray as xr
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

variable = 'TS'
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0.{variable}.188001-200512'
path_to = join('data', f"{model.format('')}")
fn = join(path_to, 'TS.nc')
data = xr.open_dataset(fn)[variable]

m = data.mean(dim=['lat', 'lon'])

m.plot(row='model', x='time')
plt.show()

y = m.to_numpy()

scaler = StandardScaler()
scaler.fit(y)
z = scaler.transform(y)

pca = PCA(n_components=10)
pca.fit(z)
x_new = pca.transform(z)
for i in pca.explained_variance_ratio_ * 100:
    print(i)