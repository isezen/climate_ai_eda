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
# import calendar as cal
from os.path import join
from os import makedirs as mkdir
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pickle as pkl
import scipy
from scipy.sparse import lil_matrix


def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    arr = lil_matrix(df.shape, dtype=np.uint8)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()


def get_csr_memory_usage(matrix):
    BYTES_TO_MB_DIV = 0.000001
    mem = (X_csr.data.nbytes + X_csr.indptr.nbytes +
           X_csr.indices.nbytes) * BYTES_TO_MB_DIV
    print("Memory usage is " + str(mem) + " MB")


encoding = {'dtype': np.dtype('float32'), 'zlib': True, 'complevel': 5}
val_indices = np.arange(1306, 1406)
model = 'f.e11.FAMIPC5CN'
# data_path = f'/mnt/data/CESM1/{model}'
data_path = f'/home/isezen/{model}'

out_path = join(data_path, 'output')
in_path = join(data_path, 'input')
sim_path = join(out_path, 'sim0')

data = xr.open_dataset(join(in_path, 'VARS.nc'))['TS']
data.load()
dates = pd.to_datetime(data.coords['time'].values)
years = dates.year.values.tolist()
months = dates.month.values.tolist()
data = data.assign_coords({'year': ('time', years),
                           'month': ('time', months)})
model_names = data.coords['model'].values.tolist()


itime = np.arange(data.shape[1])
itest = np.arange(1306, 1406)
itrain = np.delete(itime, itest)

data_train = data[:, itrain, :, :]
data_test = data[:, itest, :, :]

# models = {m: None for m in model_names}
# # i, tr, ts = 1, data_train[0], data_test[0]
# for i, tr, ts in zip(range(1, 11), data_train, data_test):
#     model_name = tr.coords['model'].values.tolist()
#     months_tr = tr.coords['month'].values
#     months_ts = ts.coords['month'].values
#     print(model_name)
#     models[model_name] = {m: None for m in range(1, 13)}
#     for j in models[model_name].keys():
#         print(cal.month_abbr[j])
#         df_tr = tr[np.where(months_tr == j)].to_dataframe().reset_index()
#         df_ts = ts[np.where(months_ts == j)].to_dataframe().reset_index()
#         X = df_tr[['lat', 'lon', 'year']]
#         X = pd.get_dummies(X, columns=['lat', 'lon'], sparse=True)
#         y = df_tr['TS']
#         lm = LinearRegression(n_jobs=16, copy_X=False)
#         lm.fit(X, y)
#         models[model_name][j] = lm

# with open('lm_models_dict.pkl', "wb") as f:
#     pkl.dump(models, f)

models = {m: None for m in model_names}
# i, tr, ts = 1, data_train[0], data_test[0]
for i, tr, ts in zip(range(1, 11), data_train, data_test):
    model_name = tr.coords['model'].values.tolist()
    df_tr = tr.to_dataframe().reset_index()
    df_tr.drop(['time', 'model'], axis=1, inplace=True)
    X = pd.get_dummies(df_tr[['year', 'month', 'lat', 'lon']],
                       columns=['lat', 'lon', 'month'], sparse=True)
    print('X created')
    # X.info(memory_usage='deep')
    X_csr = data_frame_to_scipy_sparse_matrix(X)
    print('CSR created')
    # get_csr_memory_usage(X_csr)
    # X2 = scipy.sparse.csr_matrix(X.values)
    y = df_tr['TS']
    del df_tr
    print('lm started')
    lm = LinearRegression(n_jobs=48)
    lm.fit(X_csr, y)
    models[model_name] = lm
    print(model_name, 'completed')

with open('lm_models_dict.pkl', "wb") as f:
    pkl.dump(models, f)

with open('lm_models_dict.pkl', "rb") as f:
    models = pkl.load(f)

data_pred = data_test.copy()
data_pred.data = np.zeros(data_pred.shape)
preds = []
for i, ts in enumerate(data_test):
    model_name = ts.coords['model'].values.tolist()
    print(model_name)
    df_ts = ts.to_dataframe().reset_index()
    df_ts.drop(['time', 'model'], axis=1, inplace=True)
    X = pd.get_dummies(df_ts[['year', 'month', 'lat', 'lon']],
                       columns=['lat', 'lon', 'month'], sparse=True)
    X_csr = data_frame_to_scipy_sparse_matrix(X)
    lm = models[model_name]
    y = lm.predict(X_csr)
    preds.append(y.reshape(ts.shape))

data_pred.data = np.array(preds)

mkdir(sim_path, exist_ok=True)
data_pred.to_netcdf(join(sim_path, 'pred.nc'), encoding={'TS': encoding})
