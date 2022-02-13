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

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.plot(0)
    # plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

variable = 'TS'
model = f'f.e11.FAMIPC5CN.f09_f09.historical.toga{{}}.cam.h0.{variable}.188001-200512'
path_to = join('data', f"{model.format('')}")
fn = join(path_to, 'TS.nc')
data = xr.open_dataset(fn)[variable]
model_name = data.coords['model'].values.tolist()

# model_name = [f'ens{i:02d}' for i in range(1, 11)]
# ens = np.load('data/ens1-10_network_preds_val.npy')
# x = xr.DataArray(ens)
# x = x.rename({'dim_0': 'model', 'dim_1': 'time', 
#               'dim_2': 'lat', 'dim_3': 'lon'})
# x = x.assign_coords(model=model_name, time=data.coords['time'][1306:1406],
#                     lat=data.coords['lat'], lon=data.coords['lon'])

y = data.to_numpy()
z = y.reshape(10, -1).T

scaler = StandardScaler()
scaler.fit(z)
z = scaler.transform(z)

pca = PCA()
pca.fit(z)
for i in pca.explained_variance_ratio_ * 100:
    print(i)

x_new = pca.transform(z)

fig = plt.figure()
myplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]),
       model_name)
plt.tight_layout()
fig.set_size_inches(10, 10)
fig.savefig(f'figures/ensemble_pca.pdf', bbox_inches='tight')
plt.close(fig)




