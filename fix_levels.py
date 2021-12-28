#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:31:48 2021

@author: lucas

# =============================================================================
# Interpolate a raw EMEP output from sigma coordinates to pressure coordinates.
# X(time, lev, lat, lon) -> X(time, p, lat, lon)
#
# How to use from command line:
# python3 fix_levels.py <varname> <input.nc> <output.nc>
# 
# Examples with single or multiple variables:
# python3 fix_levels.py D3_ppb_O3 EMEPOUT.nc EMEPOUT_pressure.nc
# python3 fix_levels.py NO2,NO,O3 EMEPOUT.nc EMEPOUT_pressure.nc
# =============================================================================

"""

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from tqdm import trange
import sys
import gc
import os
# %%

# =============================================================================
# READING/LOADING DATA
# =============================================================================
try:
    vars_to_modify = sys.argv[1].split(",")
    data_path = sys.argv[2]
    outputpath = sys.argv[3]
except:
    vars_to_modify = ["D3_ppb_O3"]
    data_path = "datos/MAZZEO_BC12_hour.nc"
    outputpath = "emepout.nc"


print('Loading data...')
print('Input: ' + data_path)
print('Variable(s): ' + ",".join(vars_to_modify))
print('Output: ' + outputpath)
data = xr.open_dataset(data_path)
# Pressure levels (hectopascals)
pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
                   700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
                   200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 5]

print('Selected pressure levels: \n')
for p in pressure_levels:
    print(p)
lev = data.lev.values
data = data.assign_coords({'p': np.array(pressure_levels, float)})

print('Checking if PS, hyam and hybm variables exists on input !!')
# Loading Surface Pressure
try:
    PS = data['PS']
    print('Variable PS found !')
except ValueError:
    print('Surface pressure variable "PS" not found on input !!\nExit')

# Checking EMEP sigma to pressure inner variables exists!
try:
    hyam, hybm = data['hyam'], data['hybm']
    print('Variables hyam and hybm found !')
except ValueError:
    print('"hyam" and "hybm" doesnt exist on input !!\nThese variables\
          are needed to interpolate sigma levels to pressure !\nExit')

data = data[vars_to_modify+['PS','hyam','hybm']]
# Checking input coordinates and dimensions
print('Checking input coordinates and grid...')
coords_dims = dict(data.coords.dims)
if ('lon' in coords_dims.keys()) & ('lat' in coords_dims.keys()):
    print('lat/lon are valid dimension, Grids fine.')
else:
    print('lat/lon dimensions not found, trying to build them...')
    try:
        print('lat/lon variables found in netcdf...')
        lat = data.lat.values[:, 0].squeeze().astype(float)
        lon = data.lon.values[0, :].squeeze().astype(float)
        data = data.assign_coords({'lat': lat, 'lon': lon})
        del lat, lon
    except:
        raise ValueError(
            'Input netcdf doesnt have lat/lon variables nor dimensions')


coords_dims = dict(data.coords.dims)
# %%

# Get the pressure field from each sigma level

print('Computing the pressure field asigned to each sigma level...')
pressure = np.empty(data[vars_to_modify[0]].shape)
for i in range(data.lev.shape[0]):
    pressure[:, i, :, :] = PS*hybm[i]+hyam[i]
pressure = pressure.reshape((coords_dims['time'], coords_dims['lev'],
                             coords_dims['lat']*coords_dims['lon']))

print('Interpolate sigma levels to pressure levels... ')
for i in range(len(vars_to_modify)):
    var_name = vars_to_modify[i]
    var = data[var_name].values
    data = data.drop_vars(var_name)
    var = np.reshape(var, (coords_dims['time'], coords_dims['lev'],
                           coords_dims['lat']*coords_dims['lon']))

    new_var = np.empty((coords_dims['time'], len(pressure_levels),
                        coords_dims['lat']*coords_dims['lon']))
    for t in trange(new_var.shape[0]):
        for place in range(new_var.shape[2]):
            int_sigma = interp1d(pressure[t, :, place], lev,
                                 fill_value='extrapolate')
            int_func = interp1d(lev, var[t, :, place],
                                fill_value='extrapolate')
            new_var[t, :, place] = int_func(int_sigma(pressure_levels))

    new_var = new_var.reshape((coords_dims['time'], len(pressure_levels),
                              coords_dims['lat'], coords_dims['lon']))
    data[var_name] = (['time', 'p', 'lat', 'lon'],  new_var)
    np.save(vars_to_modify[i]+'.npy', new_var)
    del new_var, var_name, var
    gc.collect()

print('Saving file...')
variables = []
for i, var in enumerate(vars_to_modify):
    variables.append(np.load(var+'.npy').astype(np.float32))
    data = data.drop_vars(var)
    data[var] = (['time', 'p', 'lat', 'lon'],
                 np.where(variables[i] > 0,
                          variables[i],
                          np.nan))
del variables
data.to_netcdf(outputpath)
for var in vars_to_modify:
    os.system("rm -rf "+var+".npy")
print('Done')
