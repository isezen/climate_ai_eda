#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621,W0702,W0703

"""
Create plots for AI project NC files
~~~~~~~
Python script to create plots
"""

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import pandas as pd

def map(x, cmap='twilight'):
    p = x.plot(col="month", col_wrap=3, cmap=cmap,
               aspect=x.shape[2] / x.shape[1],
               subplot_kws={"projection": ccrs.PlateCarree()})
    for i, ax in enumerate(p.axes.flat):
        print(i)
        ax.coastlines(lw=0.1)
        if (i + 1) % 3 == 1:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--', dms=True)
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        if i == 9:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--', dms=True)
            gl.top_labels = False
            gl.bottom_labels = True
            gl.right_labels = False
            gl.left_labels = True
        elif i > 9:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--', dms=True)
            gl.top_labels = False
            gl.bottom_labels = True
            gl.right_labels = False
            gl.left_labels = False
        else:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.7, color='gray', alpha=0.5, linestyle='--', dms=True)        
    return p    

def save_map(x, fname, cmap='twilight'):
    from os import makedirs as _mkdir
    from os.path import join as _join
    pth = "figures"
    p = map(x, cmap)
    _mkdir(pth, exist_ok=True)
    p.fig.savefig(_join(pth, fname), bbox_inches='tight')
    plt.close(p.fig)

metric_names = ["ME", "MAE", "RMSE", "MAXE", "PAE"]
pred_names = ['Mean', 'AI']
real = xr.open_dataset('data/real_val.nc')['TS'] - 272.15
mean = xr.open_dataset('data/mean_val.nc')['TS'] - 272.15
pred = xr.open_dataset('data/pred_val.nc')['TS'] - 272.15


m = (mean - real) # Mean vs real
p = (pred - real) # Prediction vs real
ma, pa = abs(m), abs(p)
mp = [m, p, ma, pa]

# stats
data = [[i.min().values.tolist(), 
         i.quantile(0.05).values.tolist(),
         i.quantile(0.25).values.tolist(),
         i.quantile(0.5).values.tolist(),
         i.mean().values.tolist(),
         i.quantile(0.75).values.tolist(),
         i.quantile(0.95).values.tolist(),
         i.max().values.tolist()] for i in mp]
df_stats = pd.DataFrame(data, 
    columns = ['Min', 'Q05', 'Q25', 'Median', 'Mean', 'Q75', 'Q95', 'Max'],
    index=['Mean', 'AI', 'Mean.Abs', 'AI.abs']).T

mme = m.mean(axis=0)
mmae = ma.mean(axis=0)
mmaxe = ma.max(axis=0)
mpae = ma.quantile(0.75, dim='time')
mrmse = np.sqrt((ma**2).mean(axis=0))
mean_metrics = xr.concat([mme, mmae, mrmse, mmaxe, mpae], dim='metrics')
mean_metrics = mean_metrics.assign_coords(metrics=metric_names)

pme = p.mean(axis=0)
pmae = pa.mean(axis=0)
pmaxe = pa.max(axis=0)
ppae = pa.quantile(0.75, dim='time')
prmse = np.sqrt((pa**2).mean(axis=0))
pred_metrics = xr.concat([pme, pmae, prmse, pmaxe, ppae], dim='metrics')
pred_metrics = pred_metrics.assign_coords(metrics=metric_names)

metrics = xr.concat([mean_metrics, pred_metrics], dim = "prediction")
metrics = metrics.assign_coords(prediction=pred_names)


nc, nr = metrics.shape[0:2]
fig, axes = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(10, 12),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         gridspec_kw={'width_ratios': [1, 1.2]})
fig.text(0.5, 0.01, 'Longitude', ha='center')
fig.text(0.01, 0.5, 'Latitude', va='center', rotation='vertical')
top_title = pred_names
left_title = metric_names
for i in range(len(left_title)):
    fig.text(0.48, 1 - 0.15 - i * 0.18, left_title[i], ha='center')
for i in range(nr):
    for j in range(nc):
        dat = metrics[:,i,:,:]
        rng = [dat.min().values.tolist(), dat.max().values.tolist()]
        add_colorbar = j % nc == 1
        cmap = 'twilight_shifted' if i == 0 else 'twilight'
        ax = axes[i][j]
        pm = dat[j].plot(ax=ax, vmin=rng[0], vmax=rng[1], cmap=cmap,
                         x='lon', y='lat',
                         add_colorbar=add_colorbar)
        # ax = pm.axes
        ax.coastlines(lw=0.1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                          linewidth=0.7, color='gray', alpha=0.5,
                          linestyle='--', dms=True)
        gl.top_labels = i == 0
        gl.bottom_labels = i == nr - 1
        gl.right_labels = False
        gl.left_labels = j == 0
        tt = top_title[j] if i == 0 else ''
        lt = left_title[i] if j == 1 else ''
        xlab = 'Longitude' if i == 4 else ''
        ax.set_title(tt)
        ax.set_ylabel(lt)

fig.tight_layout(pad=2.0)
fig.savefig(f'figures/metrics.pdf')
plt.close(fig)


# ---------------------------------------------------------------------------
# monthly
mma = ma.groupby('time.month')
m_mme = m.groupby('time.month').mean()
m_mmae = mma.mean()
m_mmaxe = mma.max()
m_mpae = mma.quantile(0.75, dim="time")
m_mrmse = np.sqrt((m**2).groupby('time.month').mean())
monthly_mean_metrics = xr.concat([m_mme, m_mmae, m_mrmse, m_mmaxe, m_mpae], dim='metrics')
monthly_mean_metrics = monthly_mean_metrics.assign_coords(metrics=metric_names)


mmp = pa.groupby('time.month')
m_pme = p.groupby('time.month').mean()
m_pmae = mmp.mean()
m_pmaxe = mmp.max()
m_ppae = mmp.quantile(0.75, dim="time")
m_prmse = np.sqrt((p**2).groupby('time.month').mean())
monthly_pred_metrics = xr.concat([m_pme, m_pmae, m_prmse, m_pmaxe, m_ppae], dim='metrics')
monthly_pred_metrics = monthly_pred_metrics.assign_coords(metrics=metric_names)

monthly_metrics = xr.concat([monthly_mean_metrics, monthly_pred_metrics], dim = "prediction")
monthly_metrics = monthly_metrics.assign_coords(prediction=pred_names)

predictions = metrics.coords['prediction'].values.tolist()
metric_names = metrics.coords['metrics'].values.tolist()

for pred in monthly_metrics:
    for mn in pred:
        pred_name = mn.coords['prediction'].values.tolist()
        metric_name = mn.coords['metrics'].values.tolist()
        cmap = 'twilight_shifted' if metric_name == 'ME' else 'twilight'
        save_map(mn, f"monthly_{pred_name}_{metric_name}.pdf", cmap)


# boxplots
mdf = m.to_dataframe().reset_index()
months = [i[1][0].month for i in mdf.iterrows()]
mdf['prediction'] = 'Mean'
mdf['type'] = 'dif'
mdf['month'] = months
pdf = p.to_dataframe().reset_index()
pdf['prediction'] = 'AI'
pdf['type'] = 'dif'
pdf['month'] = months
madf = ma.to_dataframe().reset_index()
madf['prediction'] = 'Mean'
madf['type'] = 'abs'
madf['month'] = months
padf = pa.to_dataframe().reset_index()
padf['prediction'] = 'AI'
padf['type'] = 'abs'
padf['month'] = months
df = pd.concat([mdf, pdf, madf, padf])

# world
plt_bxp = sns.catplot(data=df, x='prediction', y='TS', 
        row='type', kind='box', showfliers=False)
fig = plt_bxp.figure
fig.savefig(f'figures/bxp_mean_ai.pdf', bbox_inches='tight')
plt.close(fig)

df2 = df[df['type'] == 'dif']
plt_bxp = sns.catplot(data=df2, x='prediction', y='TS', 
        col='month', col_wrap=3, kind='box', showfliers=False)
fig = plt_bxp.figure
fig.savefig(f'figures/bxp_monthly_mean_ai.pdf', bbox_inches='tight')
plt.close(fig)

df2 = df[df['type'] == 'dif']
plt_bxp = sns.catplot(data=df2, x='month', y='TS', 
        row='prediction', kind='box', showfliers=False,
        height=5, aspect=10/5)
fig = plt_bxp.figure
fig.savefig(f'figures/bxp_monthly_mean_ai_2.pdf', bbox_inches='tight')
plt.close(fig)

df2 = df[df['type'] == 'dif']
df2 = df2.reset_index()
# df2 = df2[~df2.index.duplicated()]
plt_ecdf = sns.ecdfplot(data=df2, x="TS", hue="prediction")
fig = plt_ecdf.figure
fig.savefig(f'figures/ecdf_mean_ai.pdf', bbox_inches='tight')
plt.close(fig)

# Turkey
df2 = df[(df['lat'] >= 35) & (df['lat'] <=42) & (df['lon'] >= 26) & (df['lon'] <=44)]
plt_bxp = sns.catplot(data=df2, x='prediction', y='TS', 
        row='type', kind='box', showfliers=False)
fig = plt_bxp.figure
fig.savefig(f'figures/bxp_mean_ai_turkey.pdf', bbox_inches='tight')
plt.close(fig)


# -----------
# Error by dimension
freq = int(3)
df2 = df[df['type'] == 'dif']
for d in ['lat', 'lon', 'time']:
    plt_bxp = sns.catplot(data=df2, x=d, y='TS', 
            row='prediction', kind='box', showfliers=False)
    ax = plt_bxp.axes[1][0]
    tick_labels = ax.get_xticklabels()[::freq]
    if d == 'time':
        for tl in tick_labels:
            tl.set_text(tl.get_text()[0:7])
    xtix = ax.get_xticks()
    ax.set_xticks(xtix[::freq])
    ax.set_xticklabels(tick_labels, rotation=90)
    plt.tight_layout()
    fig = plt_bxp.figure
    fig.set_size_inches(30, 12)
    fig.savefig(f'figures/bxp_{d}.pdf', bbox_inches='tight')
    plt.close(fig)

df2 = df[df['type'] == 'abs']
for d in ['lat', 'lon', 'time']:
    plt_bxp = sns.catplot(data=df2, x=d, y='TS', 
            row='prediction', kind='box', showfliers=False)
    ax = plt_bxp.axes[1][0]
    tick_labels = ax.get_xticklabels()[::freq]
    if d == 'time':
        for tl in tick_labels:
            tl.set_text(tl.get_text()[0:7])
    xtix = ax.get_xticks()
    ax.set_xticks(xtix[::freq])
    ax.set_xticklabels(tick_labels, rotation=90)
    plt.tight_layout()
    fig = plt_bxp.figure
    fig.set_size_inches(30, 30)
    fig.savefig(f'figures/bxp_{d}_abs.pdf', bbox_inches='tight')
    plt.close(fig)
