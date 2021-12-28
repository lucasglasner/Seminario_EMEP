#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:55:09 2021

@author: lucas
"""

from Info_esta import info_esta
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cf
from glob import glob
import sys
sys.path.append('datos/')

# %%

ticco = xr.open_dataset('datos/TICCO_BC24_hour_pressure.nc', chunks=None)
lat, lon = ticco.lat.values.squeeze(), ticco.lon.values.squeeze()
ticco = xr.open_dataset('datos/TICCO_SURF.nc',
                        chunks=None).drop('projection_lambert')
ticco = ticco.drop(['i', 'j']).assign_coords({'lat': lat, 'lon': lon})
for v in ticco.keys():
    ticco[v] = (['time', 'lat', 'lon'], ticco[v].values.squeeze())

del lat, lon
mazzeo = xr.open_dataset('datos/MAZZEO_SURF.nc', chunks=None)


# %%

paths = glob('datos/*.csv')
paths = paths[1:]
names = ["_".join(x.split("/")[1].split("_")[:2]) for x in paths]
data = {key: None for key in names}
for n, p in zip(names, paths):
    data[n] = pd.read_csv(p, sep=";", dtype=str).dropna(
        how='all').dropna(axis=1, how='all')
    if data[n].shape[0] == 719:
        data[n] = data[n].iloc[:-23, :]
    data[n].index = pd.date_range('2016-01-02', '2016-01-31', freq='h')[1:]
    data[n] = data[n].iloc[:, 2]

dataO3 = {}
dataNOX = {}
for n in names:
    if 'NOX' in n:
        dataNOX[n] = data[n]
    if 'O3' in n:
        dataO3[n] = data[n]
# dataNOX['LaFlorida_NOX'] = dataNOX['LaFlorida_NOX'].iloc[:-23,:]

stations = ['Colmo', 'La Florida', 'La Palma', 'Las Condes', 'Parque Ohiggins',
            'Puchuncavi', 'Pudahuel', 'Quintero', 'Valle Alegre']
st_coords = [list(info_esta[n]['cord'].values()) for n in stations]
stations = ['Colmo', 'LaFlorida', 'LaPalma', 'LasCondes', 'Ohiggins', 'Puchuncavi',
            'Pudahuel', 'Quintero', 'ValleAlegre']
st_coords = pd.DataFrame(
    st_coords, columns=['lon', 'lat'], index=stations, dtype=float)


def f(x): return float(x.replace(",", ".") if type(x) == str else float(x))


dataO3 = pd.concat(dataO3, axis=1).applymap(f)
dataO3.columns = [m.split("_")[0] for m in dataO3.columns]
dataNOX = pd.concat(dataNOX, axis=1).applymap(f)
dataNOX.columns = [m.split("_")[0] for m in dataNOX.columns]
# %%

dataO3_TICCO = {n: None for n in stations}
dataO3_MAZZEO = {n: None for n in stations}
dataNOX_TICCO = {n: None for n in stations}
dataNOX_MAZZEO = {n: None for n in stations}
for name in stations:
    coords = st_coords.loc[name]
    dataO3_TICCO[name] = ticco.SURF_ppb_O3.sel(
        lat=coords.lat, lon=coords.lon, method='nearest').to_series()
    dataO3_MAZZEO[name] = mazzeo.SURF_ppb_O3.sel(
        lat=coords.lat, lon=coords.lon, method='nearest').to_series()
    dataNOX_TICCO[name] = ticco.SURF_ppb_NO.sel(lat=coords.lat, lon=coords.lon, method='nearest').to_series(
    )+ticco.SURF_ppb_NO2.sel(lat=coords.lat, lon=coords.lon, method='nearest').to_series()
    dataNOX_MAZZEO[name] = mazzeo.SURF_ppb_NO.sel(lat=coords.lat, lon=coords.lon, method='nearest').to_series(
    )+mazzeo.SURF_ppb_NO2.sel(lat=coords.lat, lon=coords.lon, method='nearest').to_series()

dataO3_TICCO = pd.concat(dataO3_TICCO, axis=1).reindex(
    dataO3.index, method='nearest')
dataNOX_TICCO = pd.concat(dataNOX_TICCO, axis=1).reindex(
    dataO3.index, method='nearest')
dataO3_MAZZEO = pd.concat(dataO3_MAZZEO, axis=1).reindex(
    dataO3.index, method='nearest')
dataNOX_MAZZEO = pd.concat(dataNOX_MAZZEO, axis=1).reindex(
    dataO3.index, method='nearest')


# %%
metricO3 = ((dataO3_TICCO-dataO3)**2-(dataO3_MAZZEO-dataO3)**2) / \
    pd.concat([(dataO3_TICCO-dataO3)**2, (dataO3_MAZZEO-dataO3)**2]).max(level=0)
metricNOX = ((dataNOX_TICCO-dataNOX)**2-(dataNOX_MAZZEO-dataNOX)**2) / \
    pd.concat([(dataNOX_TICCO-dataNOX)**2,
              (dataNOX_MAZZEO-dataNOX)**2]).max(level=0)

# %%

fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
ax = ax.ravel()
for axis in ax:
    axis.set_extent([mazzeo.lon.min(), mazzeo.lon.max(),
                    mazzeo.lat.min(), mazzeo.lat.max()])
    axis.coastlines()
    axis.add_feature(cf.BORDERS)
    gl = axis.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
gl.left_labels = False
gl.right_labels = True

for st in stations:
    ax[0].scatter(st_coords.loc[st].lon, st_coords.loc[st].lat,
                  transform=ccrs.PlateCarree(), c=metricO3[st].mean(),
                  cmap='RdBu',vmin=-0.25,vmax=0.25, edgecolor="k")
    ax[1].scatter(st_coords.loc[st].lon, st_coords.loc[st].lat,
                  transform=ccrs.PlateCarree(), c=metricNOX[st].mean(),
                  cmap='RdBu',vmin=-0.25,vmax=0.25, edgecolor="k")
    
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=colors.Normalize(-0.25,0.25))   
    
cax = fig.add_axes([1.1,0.25,0.01,0.5])
    
ax[0].set_title('O3')
ax[1].set_title('NOX')
fig.colorbar(sm,cax,label='AV: Added Value Metric (-)')
    
plt.savefig('ganancia.jpg',dpi=150,bbox_inches="tight")
    
