#!/bin/bash
conda create --name py38 -c conda-forge python=3.8 -y
conda activate py38
pip install aind-data-transfer xarray_multiscale h5py kerchunk ujson