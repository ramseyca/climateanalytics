#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:32:30 2024

@author: climuser
"""

##cmip 6 data available: https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=form
import xarray as xr
import numpy as np
file = 'hus_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_19810116-20101216_v20190726.nc'
file2 = 'hus_Amon_GFDL-ESM4_ssp370_r1i1p1f1_gr1_20810116-21001216_v20180701.nc'
folder = '/media/sf_Documents/GEOG_4574/SP24/module_10/'
filename = folder+file
filename2 = folder+file2

latint = 37.227
lonint = -80.411

ds = xr.open_dataset(filename).metpy.parse_cf()

ds_point = ds.sel(lat=latint,lon=360+lonint,plev=925,method='nearest')

dsf = xr.open_dataset(filename2).metpy.parse_cf()

dsf_point = dsf.sel(lat=latint,lon=360+lonint,plev=925,method='nearest')

#calculate the monthly means of historical data
ds_full_mon = ds.groupby("time.month").mean(dim="time")
ds_mon = ds_point.groupby("time.month").mean(dim="time")
ds_mon_var = ds_mon.hus

#calculate the monthly means of future data
dsf_full_mon = dsf.groupby("time.month").mean(dim="time")

dsf_mon = dsf_point.groupby("time.month").mean(dim="time")
dsf_mon_var = dsf_mon.hus

#put data into a pandas dataframe for easy plotting
import pandas as pd
df = pd.DataFrame({'historical':ds_mon_var,'future':dsf_mon_var})
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(dpi=400) #dpi

g = sns.lineplot(df)
g.set_xticks(range(len(df)))
g.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'])
plt.ylabel('Specific Humidity (kg/kg)')
plt.savefig(folder+'925_sh_bburg_line.jpg')


ds_diff = dsf_full_mon - ds_full_mon

ds_diff_jul = ds_diff.sel(month=8)

hus_diff_plot = ds_diff_jul.hus.values.squeeze()

lats = ds_diff.lons.values
lons = ds_diff.lats.values

### Module 11
folder = '/media/sf_Documents/GEOG_4574/SP24/module_11/'
file = 'gfdl_hist_8114_regrid.nc'
ds_gfdl_hist = xr.open_dataset(folder+file)
file2 = 'tasmax_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19810116-20141216_v20210113.nc'
ds_ecc_hist = xr.open_dataset(folder+file2)
gfdl_seasonal_mean_hist = ds_gfdl_hist.groupby("time.season").mean(dim="time")
ecc_seasonal_mean_hist = ds_ecc_hist.groupby("time.season").mean(dim="time")

ens_hist = np.nanmean(np.array([gfdl_seasonal_mean_hist.tasmax.values,
                             ecc_seasonal_mean_hist.tasmax.values]),axis=0)
lats = ds_ecc_hist.lat.values
lons = ds_ecc_hist.lon.values

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def add_axes(fig, grid_space):
    ax = fig.add_subplot(grid_space, projection=ccrs.PlateCarree())

    coast = cfeature.NaturalEarthFeature(category='physical',scale='10m',
                                         facecolor='none',name='coastline')
    ax.add_feature(coast,edgecolor='black',zorder=10,linewidth=0.5)
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces',
        scale='10m',
        facecolor='none')
    ax.add_feature(states_provinces,edgecolor='gray',zorder=8)
    return ax
# the 4 below refers to the number of seasons
# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(16, 7), constrained_layout=True)
nplot = 4
nrow = 2
ncol = 2
# Create gridspec to hold six subplots
grid = fig.add_gridspec(ncols=ncol, nrows=nrow)

counter = 1
ax_list = list()
# Add the axes
for i in range(nrow):
    for j in range(ncol):
        locals()["ax"+str(counter)] = add_axes(fig, grid[i, j])
        ax_list.append("ax"+str(counter))
        counter = counter+1
# Set contour levels
min_contour = 270
max_contour = 315
space = 5
levels = np.arange(min_contour, max_contour, space)
colormap = cm.Reds        

for i, axes in enumerate([ax1,ax2,ax3,ax4]):#, ax7, ax8, ax9, ax10, ax11, ax12]):#, ax13, ax14, ax15, ax16, ax17, ax18]):
            dataset = ens_hist[i,:,:]
            # Contourf plot data
            contour = axes.contourf(lons,
                                    lats,
                                    dataset,
                                    #vmin=levels[0],
                                    #vmax=levels[-1],
                                    cmap=colormap,
                                    levels=levels,
                                    extend='both')
            axes.set_title(str(ecc_seasonal_mean_hist.season[i].values),size=16)

# Set colorbounds of norm
colorbounds = np.arange(min_contour, max_contour, space)
# Use cmap to create a norm and mappable for colorbar to be correctly plotted
norm = mcolors.BoundaryNorm(colorbounds, colormap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=colormap)

# Add colorbar for all six plots
cbar = fig.colorbar(mappable,
             ax=[ax1,ax2,ax3,ax4],#, ax7, ax8, ax9, ax10, ax11, ax12], #ax13, ax14, ax15, ax16, ax17, ax18],
             ticks=colorbounds,
             drawedges=True,
             orientation='horizontal',
             shrink=.8,
             pad=0.01,
             aspect=35,
             extendfrac='auto',
             extendrect=True,)
cbar.ax.tick_params(labelsize = 18)
cbar.set_label('Precipitation (mm/day)',size=18)
#fig.set_constrained_layout_pads(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
#                                wspace=-0.5)#wspace=0 for 18
# Add figure titles
#fig.suptitle("rectilinear_grid_2D.nc", fontsize=22, fontweight='bold')
#ax1.set_title("surface temperature", loc="left", fontsize=16, y=1.05)
#ax2.set_title("degK", loc="right", fontsize=15, y=1.05)
plt.savefig(folder+'module11_hist_panel'+'.jpg',dpi=1000)
# Show plot
plt.show()

    



