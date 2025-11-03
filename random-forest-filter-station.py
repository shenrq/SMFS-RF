#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR: Shen Ruoque
VERSION: v2025.10.31

RF model training
"""
#%%
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
import os
import re
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from scipy.stats import norm

#%%
def read_curves(file, croptype):
    """
    Load NDVI time series of crop pixels around an AMS station.

    return times series (46 × N) and N (number of crop pixels)
    """
    curves = sio.loadmat(os.path.join(path, croptype, file))["arr_0"]
    return curves, curves.shape[1]

def read_offsets(file, croptype, number):
    """
    Read offsets (difference between SMF-S-extracted and observed phenological dates) from .npz files.
    """
    offsets = {}
    for pheno in phenos:
        file2 = os.path.join(
            path, croptype, re.sub("NDVI-time_series", f"y_offset-SMF-S-{pheno}-w{w}", file)
        )[:-4] + ".npz"
        if not os.path.isfile(file2):
            offset = np.ones(number) * np.nan
        else:
            offset = np.load(os.path.join(path, croptype, file2))["arr_0"]
        offsets[pheno] = offset
    return offsets

def read_curves_offsets(file, croptype):
    """
    Read NDVI time series and offsets.
    """
    curves, number = read_curves(file, croptype)
    offsets = read_offsets(file, croptype, number)
    return curves, offsets, number

def filter_curves(curves, offsets, number, pixlim):
    """
    Randomly select `pixlim` time series from all NDVI time series.
    """
    if number < pixlim:
        return curves, number, offsets
    select = np.random.choice(number, size=pixlim, replace=False)
    for pheno in phenos:
        offsets[pheno] = offsets[pheno][select]
    return curves[:, select], pixlim, offsets

def get_label(y0, nums):
    """
    observation, offset -> training label (SMF-S estimated phenological dates)
    """
    y = np.zeros([np.sum(nums)])
    n = 0
    for i, num in enumerate(nums):
        y[n:n+num] = y0[i]
        n += num
    return y

def get_station(file):
    """
    return station ID and observed year.
    """
    pattern = re.compile(r"(\d+)-(\d{4})-NDVI-time_series-v1-buffer-0.05.mat")
    matched = pattern.match(file)
    return int(matched[1]), int(matched[2])

def get_sample_p(y0, outlier):
    """
    probability for observation sampling
    """
    mu = np.mean(y0)
    sigma = np.std(y0)
    y1 = np.minimum(y0, mu+outlier*sigma)
    y1 = np.maximum(y1, mu-outlier*sigma)
    p = 1 / norm.pdf(y1, mu, sigma)
    p = p / np.sum(p)
    return p

#%%
def SMFS_RF(sheet, pixlim, outlier, croptype, pheno):
    """
    read observations, training labels, train RF model.
    """
    sheet = sheet.copy()
    sheet.reset_index(drop=True, inplace=True)
    flist = os.listdir(os.path.join(path, croptype[0]))

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
    y0 = y0[~np.isnan(y0)]

    f_train0, _, y0_train0, _ = train_test_split(
        flist3, y0, test_size=0.3, random_state=1234
    )
    p = get_sample_p(y0_train0, outlier)
    itrain = np.random.choice(
        range(y0_train0.size), int(np.round(y0_train0.size*3)), p=p
    )
    f_train = [f_train0[i] for i in itrain]
    y0_train = y0_train0[itrain]

    modelfile = (
        f"./{croptype[1]}-{pheno}-SMF-S-w{w}-pix{pixlim}-outlier{outlier}.joblib.z"
    )
    curves_train = [filter_curves(*read_curves_offsets(file, croptype[0]), pixlim) for file in f_train]
    X_train = np.concatenate([curve[0] for curve in curves_train], axis=1)
    nums = [curve[1] for curve in curves_train]
    y_train = get_label(y0_train, nums) + np.concatenate([curve[2][pheno] for curve in curves_train])
    y_valid = ~np.isnan(y_train)
    y_train = y_train[y_valid]
    X_train = X_train[:, y_valid]

    model = RandomForestRegressor()
    model.fit(X_train.T, y_train)
    joblib.dump(model, modelfile, compress=9)
    return modelfile

#%%
sheet = pd.read_file("./single-rice-phenology.xlsx") # observed phenological dates from AMSs
# | station | lat  | lon    | altitude | PAC    | province     | year | transplanting | ... | maturity |
# | ------- | ---- | ------ | -------- | ------ | ------------ | ---- | ------------- | ... | -------- |
# | XXXXX   | XX.X | 1XX.XX | XX.X     | 230000 | Heilongjiang | 2001 | 142           | ... | 259      |
# ...

path = "./sites-curve"

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

outlier = 2
w = 8
pixlim = 100

for pheno in phenos:
    SMFS_RF(sheet, pixlim, outlier, "single", pheno)

