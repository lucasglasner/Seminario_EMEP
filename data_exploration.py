#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:26:21 2021

@author: lucas
"""

import matplotlib.colors as colors
import cartopy.crs as ccrs
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import regionmask
from shapely.ops import cascaded_union
import cartopy.feature
import datetime
from metpy.calc import height_to_pressure_std, pressure_to_height_std
from metpy.units import units


# %%
# =============================================================================
# LOAD EMEP OUT DATA.
# =============================================================================
data = xr.open_dataset("datos/MAZZEO_BC24_hour.nc", chunks=None)
data = data[["D3_ppb_NO2", "D3_ppb_NO", "D3_ppb_O3", "HMIX"]]  # Grab vars
data = data.sortby("lat").sortby("lon")      # Order by coordinates
data = data.resample({"time": "h"}).bfill()  # Fix time coordinate



lat, lon = data.lat.values, data.lon.values
LON, LAT = np.meshgrid(lon, lat)
#%%
var = data.isel(lev=-1).groupby("time.hour").max().compute()
var1 = (var.D3_ppb_NO+var.D3_ppb_NO2).mean(dim='hour')
var2 = (var.D3_ppb_O3).mean(dim='hour')


# %%

fig, ax = plt.subplots(1, 2, figsize=(
    10, 3), subplot_kw={'projection': ccrs.Mercator()})
plt.rc('font',size=15)
for axis in ax:
    axis.coastlines()
    axis.set_extent([LON.min(),LON.max(),LAT.min(),LAT.max()])
    gl=axis.gridlines(draw_labels=True, linestyle=":")
    gl.xlocator = mticker.FixedLocator([-71.7,-71.1,-70.5,-70.0])
    gl.right_labels = False
    gl.top_labels = False
gl.left_labels=False    


m1 = ax[0].pcolormesh(LON,LAT,var1.values.squeeze(), rasterized=True, shading='auto',
                      cmap='Blues', norm=colors.LogNorm(1, 100),
                      transform=ccrs.PlateCarree())
ax[0].set_title('(a)',loc='left',fontsize=18)


m2 = ax[1].pcolormesh(LON, LAT, var2.values.squeeze(), rasterized=True,
                      shading='auto', transform=ccrs.PlateCarree(),
                      cmap='Purples', norm=colors.Normalize(0, 100))

ax[1].set_title('(b)',loc='left',fontsize=18)
fig.colorbar(m1,ax=ax[0], label="Máxima razón de\nmezcla de NOx (ppb)")
fig.colorbar(m2,ax=ax[1], label="Máxima razón de\nmezcla de O3 (ppb)")

plt.savefig('plots/maximums.pdf',dpi=150,bbox_inches='tight')
# %%
# Compute heights from pressure using standard atmosphere
heights = pressure_to_height_std(data.p.values*units("hPa")).magnitude*1e3
# Assing height as an aviable vertical coordinate
data = data.assign_coords({'z': heights})

# Grab lat/lon grid
lat, lon = data.lat.values, data.lon.values
LON, LAT = np.meshgrid(lon, lat)

# Compute daily cycle
dailycycle = data.groupby("time.hour").mean().compute()
# %%
# Load Santiago polygon
santiago = gpd.read_file("datos/division_comunal.shp")
santiago = santiago[santiago["NOM_REG"] == "Región Metropolitana de Santiago"]
santiago = santiago[santiago["NOM_PROV"] == "Santiago"]
santiago = santiago[santiago["SHAPE_Area"] < 1e8]
santiago = santiago.to_crs("epsg:4326")
santiago = gpd.GeoSeries(cascaded_union(santiago.geometry), crs=santiago.crs)

# Grab the model PBL on the DGF roof
PBL_techodgf = data.HMIX.sel(lat=-33.457, lon=-70.661,
                             method="nearest").to_series()

# PBL Observations
PBL_munoz = pd.read_csv("datos/Capa Limite/mhDGF_2015a2016_fix.csv",
                        header=None, index_col=0).squeeze()
PBL_munoz.index = pd.to_datetime(PBL_munoz.index)

# %%
# Load topography data
dem = xr.open_dataset("datos/DEM_RM.nc").Band1
dem = dem.sel(lat=slice(*[lat.min(), lat.max()]),
              lon=slice(*[lon.min(), lon.max()]))
dem = dem.reindex({'lat': data.lat, 'lon': data.lon},
                  method='nearest')


# shift times
dailycycle = dailycycle.roll({'hour': 13})


# %%

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))

topo = dem.sel(lat=-33.457, method="nearest").values
press = np.array([1000, 750, 500, 250, 150])
heights = pressure_to_height_std(press*units('hPa')).magnitude
LON, LEV = np.meshgrid(data.lon, data.z)


var = dailycycle.D3_ppb_O3.max(dim="hour").sel(lat=-33.457, method="nearest")
mapa1 = ax[0].pcolormesh(LON, LEV, var,
                         cmap='Purples', norm=colors.Normalize(0, 80),
                         shading='gouraud',
                         rasterized=True)
ax[0].set_title('Maximum Ozone\nMixing Ratio (ppb)', loc='left')


var = dailycycle.D3_ppb_NO.max(dim="hour")+data.D3_ppb_NO2.max(dim='time')
var = var.sel(lat=-33.457, method="nearest")
mapa2 = ax[1].pcolormesh(LON, LEV, var,
                         cmap='Blues', norm=colors.LogNorm(1, 10), shading='gouraud',
                         rasterized=True)
ax[1].set_title('Maximum NOx\nMixing Ratio (ppb)', loc='left')

fig.colorbar(mapa1, ax=ax[0], pad=0.16)
fig.colorbar(mapa2, ax=ax[1], pad=0.16)


ax = ax.ravel()
for axis in ax:
    axis.fill_between(dem.lon, topo, np.zeros(len(topo)),
                      zorder=100, color="dimgray")
    axis.set_ylim((0, 1.5e4))
    axis.set_yticks([0, 2500, 5000, 7500, 10e3, 12.5e3, 15e3])
    axis.set_yticklabels([0, 2.5, 5.0, 7.5, 10, 12.5, 15.0])
    axis.set_ylabel("")
    axis.set_xticks([-72, -71.5, -71, -70.5, -70])
    axis.tick_params(axis="x", rotation=45)

ax3 = axis.twinx()
ax3.set_ylim(0, 1.5e4)
ax3.set_yticks(heights*1e3)
ax3.set_yticklabels(press)
ax3.set_ylabel('Pressure (hPa)')
ax[0].set_ylabel('Hieght Above Sea Level (km)')
plt.savefig('plots/maximum_profiles_santiago_TICCO.pdf',
            dpi=150, bbox_inches='tight')

# %%
# =============================================================================
# make daily cycle  var-height maps on santiago
# =============================================================================

# variable to plot (O3 or NOx)
# var = dailycycle.D3_ppb_O3.sel(lat=-33.457, method="nearest")
var = dailycycle.D3_ppb_NO.sel(lat=-33.457, method="nearest") + \
    dailycycle.D3_ppb_NO2.sel(lat=-33.457, method="nearest")


# topography
topo = dem.sel(lat=-33.457, method="nearest").values

# boundary layer
# hmix = dailycycle.HMIX.sel(lat=-33.457, method="nearest")

# color limits
vmin = 1
vmax = 10
# vmin = 0
# vmax = 80

# n = 10

# pressure to show on plot
# press = np.array([1000, 750, 500, 250, 150])
press = np.array([1000, 750, 500])
# transform pressure to height
heights = pressure_to_height_std(press*units('hPa')).magnitude

# create figure and axes
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 4))

ax = ax.ravel()  # flatten array
LON, LEV = np.meshgrid(data.lon, data.z)  # create lon/height grid

# create title of each subplot (local time utc-4)
titles = var[::3, :, :].hour.values-4
titles = list(map(lambda x: str(x+24) if x < 0 else str(x), titles))
titles = list(map(lambda x: "0"+x+":00" if len(x) < 2 else x+":00", titles))

# loop over different axes/subplots
for i, axis in enumerate(ax):
    # plot data every 3 hours
    mapa = axis.pcolormesh(LON, LEV, var[::3, :, :][i, :, :].values,
                           cmap="Blues",  rasterized=True, shading='gouraud',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    # plot mean boundary layer height
    # axis.plot(hmix.lon,
    # (hmix.values[::3, :][i, :]),
    # color="k")
    # plot topography as a shaded contour
    axis.fill_between(dem.lon, topo, np.zeros(len(topo)),
                      zorder=100, color="dimgray")
    axis.set_ylim((0, 0.75e4))  # set y limits
    axis.set_yticks([0, 2500, 5000, 7500])  # set yticks
    # axis.set_yticks([0, 2500, 5000, 7500, 10e3, 12.5e3, 15e3])  # set yticks
    # set labels for y axis
    # axis.set_yticklabels([0, 2.5, 5.0, 7.5, 10, 12.5, 15.0])
    axis.set_yticklabels([0, 2.5, 5.0, 7.5])
    axis.set_ylabel("")  # delete y label
    axis.set_title(titles[i], fontsize=10, loc="left")  # set titles
    axis.set_xticks([-71.5, -71, -70.5, -70])  # set x axis tticks
    axis.tick_params(axis="x", rotation=45)  # rotate x axis a little
# a legend for the PBL line
# ax[0].plot([], [], color="k", label="EMEP PBL\nHeight")
# ax[0].legend(frameon=False, loc="upper left")  # show legend

# make some parallel y axis for showing pressure levels
ax2 = ax[3].twinx()
ax3 = ax[-1].twinx()

# set limits of these new axes
for axis in (ax2, ax3):
    axis.set_ylim(0, 0.75e4)
    axis.set_yticks(heights*1e3)
    axis.set_yticklabels(press)

# some text and labels
fig.text(0.29, 1, 'Sección Transversal Santiago ($\phi$: 33.457°S)',
         ha='center', va='center', fontsize=13)
fig.text(0.06, 0.5, 'Altura sobre el nivel del mar (km)', ha='center', va='center',
         rotation=90)
fig.text(0.96, 0.5, 'Presión (hPa)', ha='center', va='center',
         rotation=90)
fig.text(0.5, -0.03, 'Longitud (°W)', ha='center', va='center')

# create colorbar
box1, box2 = ax[3].get_position(), ax[-1].get_position()
cax = fig.add_axes([box1.xmax*1.1, box2.ymin, 0.02, box1.ymax-box2.ymin])
fig.colorbar(mapa, cax=cax, label="Razon de Mezcla NOx (ppb)")

# savefigure
plt.savefig('plots/NOx_profile_Santiago.pdf',
            dpi=150, bbox_inches='tight')

# %%
# =============================================================================
# make daily cycle planetary boundary layer height maps
# =============================================================================

lat, lon = data.lat.values, data.lon.values
LON, LAT = np.meshgrid(lon, lat)
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 4),
                       subplot_kw={"projection": ccrs.PlateCarree()})

ax = ax.ravel()
titles = dailycycle.HMIX[::3, :, :].hour.values-4
titles = list(map(lambda x: str(x+24) if x < 0 else str(x), titles))
titles = list(map(lambda x: "0"+x+":00" if len(x) < 2 else x+":00", titles))
for i, axis in enumerate(ax):
    mapa = axis.pcolormesh(LON, LAT, dailycycle.HMIX[::3, :, :][i, :, :].values,
                           transform=ccrs.PlateCarree(), cmap="cividis",
                           vmin=0, vmax=1.5e3, rasterized=True)

    santiago.boundary.plot(ax=axis, transform=ccrs.PlateCarree(), lw=1,
                           color="tab:red", alpha=0.5)
    axis.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])

    axis.set_title(titles[i], fontsize=10, loc="left")
    axis.coastlines()
    axis.add_feature(cartopy.feature.BORDERS)
    gl = axis.gridlines(linewidth=0.5, linestyle=":")
    gl.xlocator = mticker.FixedLocator([-71.7, -70.8, -70])
    gl.ylocator = mticker.FixedLocator([-34, -33.5, -33])


for axis in (ax[0], ax[4]):
    gl = axis.gridlines(linewidth=0.0, draw_labels=True)
    gl.ylocator = mticker.FixedLocator([-34, -33.5, -33])
    # gl.xlines=False
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = False


for axis in (ax[4:]):
    gl = axis.gridlines(linewidth=0.0, draw_labels=True)
    # gl.ylines = False
    gl.xlocator = mticker.FixedLocator([-71.7, -70.8, -70])
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False


ax[0].plot([], [], color="tab:red", label="Santiago")
ax[0].legend(loc="lower left", frameon=False, fontsize=8)

box1, box2 = ax[3].get_position(), ax[-1].get_position()
cax = fig.add_axes([box1.xmax*1.05, box2.ymin, 0.02, box1.ymax-box2.ymin])
fig.colorbar(mapa, cax=cax, label="Altura promedio de\nla Capa de Mezcla (m)")

plt.savefig("plots/PBL_dailycycle.pdf", dpi=150, bbox_inches="tight")

# %%


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plt.rc('font',size=15)
fig.tight_layout(pad=3)
ax = list(ax.ravel())
ts = PBL_techodgf.copy()
ts.index = ts.index-datetime.timedelta(hours=4)
ts.plot(ax=ax[0], label="EMEP")
PBL_munoz.reindex(ts.index).plot(
    ax=ax[0], c="tab:red", label="DGF-Nefobasimetro")
ax[0].set_xlabel("")
ax[0].set_ylabel("Altura de la\ncapa de mezcla (m)")
ax[0].legend(frameon=False)

ts = ts.reindex(PBL_munoz.dropna().index).dropna()
ts2 = PBL_munoz.reindex(ts.index).dropna()
# ax[1].sharey(ax[0])
ax[1].scatter(ts, ts2, ec="k")
ax[1].plot([1e2, 1.3e3], [1e2, 1.3e3], c="k", ls="--")
ax[1].set_xlabel("EMEP Capa de Mezcla (m)")
ax[1].set_ylabel("DGF-Nefobasimetro\nCapa de Mezcla(m)")
ax[1].set_title("70.661°W ; 33.457°S", loc="left")
ax[1].set_xticks(ax[1].get_yticks())
ax[1].set_xlim(2e2, 1.3e3)
ax[1].set_ylim(2e2, 1.3e3)


text = 'Pearson $r$: '+'{:.2f}'.format(np.corrcoef(ts, ts2)[0, 1])+'\n'
text = text + 'Sesgo: ' + '{:.2f}'.format(sum(ts-ts2)/np.max((sum(ts), sum(ts2))))+'\n'
text = text + '$\sigma_{EMEP}: $'+'{:.2f}'.format(np.std(ts))+'\n'
text = text + '$\sigma_{DGF}: $'+'{:.2f}'.format(np.std(ts2))+'\n'

ax[1].text(440, 1000, text, size=12,
           va="center", ha="center", multialignment="left",
           bbox=dict(fc="none"))

plt.savefig("plots/SantiagoPBL_timeseries.pdf", dpi=150, bbox_inches="tight")
