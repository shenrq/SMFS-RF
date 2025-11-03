#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR: Shen Ruoque
VERSION: v2025.10.31

Extract NDVI time series of crop pixels around AMS station from the InENVI product
"""
#%%
import os
import numpy as np
from rioxarray import open_rasterio
import glob
import pandas as pd
import re
import scipy.io as sio
#%%
ricetype = "single", 1 # 1 for single-season rice, 2 for double- in the base map
obs = pd.read_file(f"./{ricetype[0]}-rice-phenology.xlsx") # observed phenological dates from AMSs
# | station | lat  | lon    | altitude | PAC    | province     | year | transplanting | ... | maturity |
# | ------- | ---- | ------ | -------- | ------ | ------------ | ---- | ------------- | ... | -------- |
# | XXXXX   | XX.X | 1XX.XX | XX.X     | 230000 | Heilongjiang | 2001 | 142           | ... | 259      |
# ...

sites = obs["station"].drop_duplicates().to_list()

buffer = 0.05 # degree
ndvipath = "/path/to/InENVI/product"
croppath = "/path/to/CCD-Rice/product"
outpath = f"./sites-curve/{ricetype[1]}"
res30 = 0.000269494585236 # degree, 30 m in equator

for site in sites:
    for yr in range(2001, 2015+1):
        row = obs[(obs["station"] == site) & (obs["year"] == yr)]
        if len(row) == 0: continue
        province = re.sub(r" ", "_", row["province"].to_list()[0]) # "Inner Mongolia" -> "Inner_Mongolia"
        cropfile = f"{croppath}/{yr}/CCD-Rice-{province}-{yr}-v1.tif" # base map from CCD-Rice product
        if not os.path.isfile(cropfile): continue
        lat = row["lat"].to_list()[0]
        lon = row["lon"].to_list()[0]
        outfile = f"{outpath}/{site}-{yr}-NDVI-time_series-v1-buffer-{buffer}.mat"
        if os.path.isfile(outfile): continue

        print(site, province, yr)
        flist = glob.glob(f"{ndvipath}/{province}/{province}*_{yr}*.tif") # List of InENVI files
        if len(flist) == 0: continue
        xsel = slice(lon-buffer, lon+buffer)
        ysel = slice(lat+buffer, lat-buffer)
        x_ndvi = open_rasterio(flist[0]).sel(x=xsel, y=ysel).x.values
        y_ndvi = open_rasterio(flist[0]).sel(x=xsel, y=ysel).y.values
        ndvi = np.array([
            open_rasterio(file).sel(x=xsel, y=ysel).values[0, :, :] for file in flist
        ])

        basemap = open_rasterio(cropfile).sel(x=xsel, y=ysel)
        x_map = basemap.x.values
        y_map = basemap.y.values
        basemap = basemap.values == ricetype[2] # 1 for single-season rice, 2 for double-

        left = np.maximum(x_ndvi[0], x_map[0])
        top = np.minimum(y_ndvi[0], y_map[0])
        right = np.minimum(x_ndvi[-1], x_map[-1])
        bottom = np.maximum(y_ndvi[-1], y_map[-1])
        r_ndvi = [
            np.round((left - x_ndvi[0]) / res30).astype(np.int64),
            np.round((top - y_ndvi[0]) / -res30).astype(np.int64),
            np.round((right - x_ndvi[0]) / res30).astype(np.int64),
            np.round((bottom - y_ndvi[0]) / -res30).astype(np.int64),
        ]
        r_map = [
            np.round((left - x_map[0]) / res30).astype(np.int64),
            np.round((top - y_map[0]) / -res30).astype(np.int64),
            np.round((right - x_map[0]) / res30).astype(np.int64),
            np.round((bottom - y_map[0]) / -res30).astype(np.int64),
        ]
        ndvi = ndvi[:, r_ndvi[0]:r_ndvi[2], r_ndvi[1]:r_ndvi[3]]
        basemap = basemap[0, r_map[0]:r_map[2], r_map[1]:r_map[3]]
        index = np.nonzero(basemap)
        if index[0].size == 0: continue
        ndviseries = ndvi[:, index[0], index[1]]
        sio.savemat(outfile, {"arr_0": ndviseries}, do_compression=True)
