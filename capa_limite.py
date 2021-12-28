import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import re
from netCDF4 import Dataset
import netCDF4  as netCDF4
import os


def date_range(start_date, end_date, increment, period):
    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result

def data_lag(fechas_old,xhoras):    
    fechas_new = []
    for x in fechas_old: 
        x2 =  x +  datetime.timedelta(hours=xhoras)
        fechas_new.append(x2)
    return fechas_new 

def near(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx






######################## Carga EMEP #########################################
HOME = os.chdir('/home/menares/Escritorio/Papila_run/Capa Limite')


#dataset_hour = Dataset("V_F_Base_day.nc")   # I_BBase_F_day   I_F3_Base_day.nc V_F_Base_day.nc
dataset_hour = Dataset("FULL_jan_hour.nc")   # I_BBase_F_day   I_F3_Base_day.nc V_F_Base_day.nc

print (dataset_hour.variables.keys())



cl_data = dataset_hour.variables['HMIX'][:,:,:]


time2 = dataset_hour.variables['time']
lat = dataset_hour.variables['lat'][:]
lon = dataset_hour.variables['lon'][:]
lons,lats= np.meshgrid(lon,lat)

time_convert = netCDF4.num2date(time2[:], time2.units)
inicio = time_convert[0]  -  datetime.timedelta(minutes=30)
fin    = time_convert[-1]  -datetime.timedelta(minutes=30)

UTC = -4 
fechas_emep = date_range(inicio+ datetime.timedelta(hours=UTC) , fin+ datetime.timedelta(hours=UTC)  , 1, 'hours')   

t_lon = lon
t_lat = lat

ix = near(t_lon,-70.661)
iy = near(t_lat,-33.464)

cl_parque = cl_data[:,ix,iy] 
plt.plot(fechas_emep,cl_parque,'-o')
plt.xlabel('horas')
plt.ylabel('Altura capa de mezcla [m]')



######################## Carga CL muñoz et al #########################################

data_ricardo = pd.read_csv('mhDGF_2015a2016.csv', sep=",", decimal=".", header=None)
inicio_d = datetime.datetime(2015,1,1,1) 
fin_d    = datetime.datetime(2016,12,31,23) + datetime.timedelta(hours=1)
data_fechas = date_range(inicio_d, fin_d, 1, 'hours')
cl_ricardo= data_ricardo.values[:,2]




plt.figure()
plt.plot(fechas_emep,cl_parque,'-k')
plt.xlabel('horas')
plt.ylabel('Altura capa de mezcla [m]')
plt.plot(data_fechas[data_fechas.index(inicio):data_fechas.index(fin)+1],cl_ricardo[data_fechas.index(inicio):data_fechas.index(fin)+1],'o')
plt.legend(['EMEP','Estimacion de Obs '])

plt.plot(data_fechas,cl_ricardo,'o')


