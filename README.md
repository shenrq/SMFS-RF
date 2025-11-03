# SMFS-RF: a knowledge-guided machine-learning method for crop phenology extraction from fine-resolution vegetation index data

This repository stores the codes of SMFS-RF: a knowledge-guided machine-learning method for crop phenology extraction from fine-resolution vegetation index data.


## Usage

Run scripts in the following sequence.

1. `extract-sites-crop-NDVI.py`: extract NDVI time series of crop pixels around AMS stations from the InENVI product.
2. `site-curve-offset-SMF-S.py`: estimate crop phenological dates of pixels around AMS stations using SMF-S.
3. `random-forest-filter-station.py`: train RF models.
