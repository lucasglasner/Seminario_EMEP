#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:08:18 2021

@author: lucas
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from metpy.calc import pressure_to_height_std
from metpy.units import units
# %%

mazzeo = xr.open_dataset('datos/MAZZEO_BC24_hour_pressure.nc', chunks=None)
lat, lon = mazzeo.lat.values, mazzeo.lon.values
mazzeo = mazzeo[['D3_ppb_O3', 'D3_ppb_NO', 'D3_ppb_NO2']]
mazzeo = mazzeo.sel(lat=-33.457, method="nearest")
mazzeo['D3_ppb_NOx'] = (['time', 'p', 'lon'],
                        mazzeo.D3_ppb_NO2.data+mazzeo.D3_ppb_NO.data)


ticco = xr.open_dataset('datos/TICCO_BC24_hour_pressure3.nc', chunks=None)
ticco = ticco[['D3_ppb_O3', 'D3_ppb_NO', 'D3_ppb_NO2']]
ticco = ticco.sel(lat=-33.457, method="nearest")

ticco['D3_ppb_NOx'] = (['time', 'p', 'lon'],
                       ticco.D3_ppb_NO2.data+ticco.D3_ppb_NO.data)


dem = xr.open_dataset("datos/DEM_RM.nc").Band1
dem = dem.sel(lat=slice(*[lat.min(), lat.max()]),
              lon=slice(*[lon.min(), lon.max()]))
dem = dem.sel(lat=-33.457, method='nearest')
# %%

mazzeo_cycle = mazzeo.groupby('time.hour').mean()
ticco_cycle = ticco.groupby('time.hour').mean()
v = (ticco_cycle-mazzeo_cycle)

# %%
var = v.D3_ppb_NOx.roll({'hour': 13})

vmin, vmax = -5, 5

press = np.array([1000, 750, 500, 250, 150])
heights = pressure_to_height_std(press*units('hPa')).magnitude

fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 4))
# plt.gca().invert_yaxis()
# plt.xticks(rotation=45)
ax = ax.ravel()
LON, LEV = np.meshgrid(mazzeo.lon,
                       pressure_to_height_std(mazzeo.p.values*units('hPa')).magnitude)
titles = var[::3, :, :].hour.values-4
titles = list(map(lambda x: str(x+24) if x < 0 else str(x), titles))
titles = list(map(lambda x: "0"+x+":00" if len(x) < 2 else x+":00", titles))
for i, axis in enumerate(ax):
    mapa = axis.pcolormesh(LON, LEV*1e3, var[::3, :, :][i, :, :].values,
                           cmap="RdBu_r",  rasterized=True, shading='gouraud',
                           norm=colors.Normalize(vmin=vmin, vmax=vmax))
    # mapa = axis.contourf(LON, LEV*1e3, var[::3, :, :][i, :, :].values,
    #                        cmap="RdBu", levels=10,
    #                        norm=colors.Normalize(vmin=vmin, vmax=vmax))
    axis.fill_between(dem.lon, dem, np.zeros(len(dem)),
                      zorder=100, color="dimgray")
    axis.set_ylim((0, 1.5e4))
    axis.set_yticks([0, 2500, 5000, 7500, 10e3, 12.5e3, 15e3])
    axis.set_yticklabels([0, 2.5, 5.0, 7.5, 10, 12.5, 15.0])
    axis.set_ylabel("")
    axis.set_title(titles[i], fontsize=10, loc="left")
    axis.set_xticks([-72, -71.5, -71, -70.5, -70])
    axis.tick_params(axis="x", rotation=45)
# ax[0].legend(frameon=False, loc="upper left")

ax2 = ax[3].twinx()
ax3 = ax[-1].twinx()

for axis in (ax2, ax3):
    axis.set_ylim(0, 1.5e4)
    axis.set_yticks(heights*1e3)
    axis.set_yticklabels(press)

fig.text(0.46, 1, 'Santiago Cross-Section ($\phi$: 33.457°S); 10x10km - 2x2 km Grids Difference.\n',
         ha='center', va='center', fontsize=13)
fig.text(0.06, 0.5, 'Height Above Sea Level (km)', ha='center', va='center',
         rotation=90)
fig.text(0.96, 0.5, 'Pressure (hPa)', ha='center', va='center',
         rotation=90)
fig.text(0.5, -0.03, 'Longitude (°W)', ha='center', va='center')

box1, box2 = ax[3].get_position(), ax[-1].get_position()
cax = fig.add_axes([box1.xmax*1.1, box2.ymin, 0.02, box1.ymax-box2.ymin])
fig.colorbar(mapa, cax=cax, label="NOx Mixing Ratio (ppb)")

plt.savefig('plots/NOx_profile_Santiago_difference.pdf',
            dpi=150, bbox_inches='tight')
