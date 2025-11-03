#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR: Shen Ruoque
VERSION: v2025.10.31

Using SMF-S method to estimate crop phenological dates of pixels around AMS stations.
"""
#%%
import re
import numpy as np
import os
import scipy.io as sio
import multiprocessing as mp
import pandas as pd
import sys
sys.path.append("./SMF_S_Release/src")
from smf_s_class import SMFS
#%%
def read_curves(file):
    """
    Load NDVI time series of crop pixels around an AMS station.

    return times series (46 × N) and N (number of crop pixels)
    """
    curves = sio.loadmat(os.path.join(path, crop, file))["arr_0"]
    return curves, curves.shape[1]

def cut_X(X):
    """
    Set the NDVI values outside the growing season to a single value.
    """
    if crop == "single":
        X[:max(0, int(9-w/8))] = X[max(0, int(9-w/8))]
    elif crop == "early":
        X[min(46, int(29+w/8)):] = X[min(46, int(29+w/8))]
    elif crop == "late":
        X[:max(0, int(22-w/8))] = X[max(0, int(22-w/8))]
    return X

def get_offset(curves, number, y):
    """
    Extract phenological dates using SMF-S.

    return offsets (difference between extracted and observed phenological dates).
    """
    if number == 1: return np.array([0])

    smfs_model = SMFS(np.mean(curves, axis=1), y, doys)
    smfs_model = cut_X(smfs_model)
    smfs_model.WIN = w
    offset = np.array(
        [smfs_model.doit(cut_X(curves[:, j])) for j in range(curves.shape[1])]
    )
    offset[offset == 0] = np.nan
    offset -= y
    return offset

def write_offset(file_y):
    """
    Read NDVI time series, extract phenological dates, write into an .npz file.
    """
    file, y = file_y
    outfile = os.path.join(
        path, crop, re.sub("NDVI-time_series", f"y_offset-SMF-S-{pheno}-w{w}", file)
    )[:-4]
    if os.path.isfile(outfile + ".npz"): return 1
    offset = get_offset(*read_curves(file), y)
    np.savez_compressed(outfile, offset)
    return 0

#%%
daystart = 1
dayend = 361
daystep = 8
doys = np.arange(daystart, dayend+daystep, daystep) # Days of Year
w = 8

path = "./sites-curve"
crop = "single"
sheet0 = pd.read_file(f"./{crop}-rice-phenology.xlsx") # observed phenological dates from AMSs
# | station | lat  | lon    | altitude | PAC    | province     | year | transplanting | ... | maturity |
# | ------- | ---- | ------ | -------- | ------ | ------------ | ---- | ------------- | ... | -------- |
# | XXXXX   | XX.X | 1XX.XX | XX.X     | 230000 | Heilongjiang | 2001 | 142           | ... | 259      |
# ...

phenos = [
    "transplanting",
    "regreening",
    "tillering",
    "stem_elongation",
    "booting",
    "heading",
    "milk_ripening",
    "maturity",
]

for pheno in phenos:
    sheet = sheet0.copy()
    sheet.reset_index(drop=True, inplace=True)
    flist = os.listdir(os.path.join(path, crop))

    flist2 = []
    deletion = []
    for i, row in sheet.iterrows():
        station = row["station"]
        yr = row["year"]
        file = f"{station}-{yr}-NDVI-time_series-v1-buffer-0.05.mat"
        if file in flist:
            flist2.append(file)
        else:
            deletion.append(i)
    sheet.drop(deletion, inplace=True)
    sheet.reset_index(drop=True, inplace=True)

    y0 = sheet[pheno].to_numpy()
    flist3 = [file for i, file in enumerate(flist2) if not np.isnan(y0[i])]
    y0 = list(y0[~np.isnan(y0)])

    pools = mp.Pool(processes=60)
    status = pools.map(write_offset, zip(flist3, y0))
    pools.close()
    pools.join()
