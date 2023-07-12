#!/bin/bash
conda create --name py38 -c conda-forge python=3.8 -y
conda run -n py38 pip install aind-data-transfer ome-zarr xarray_multiscale hdf5plugin kerchunk ujson tensorstore