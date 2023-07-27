#!/bin/bash
conda create --name py310 -c conda-forge python=3.10 -y
conda run -n py310 pip install git+https://github.com/jwong-nd/sofima.git@feat-zarr-dataset-creation
conda run -n py310 pip install --upgrade "jax[cpu]"
conda run -n py310 pip install tensorstore
conda run -n py310 pip install tensorflow-cpu
conda run -n py310 pip install aind-data-transfer ome-zarr xarray_multiscale hdf5plugin kerchunk ujson
