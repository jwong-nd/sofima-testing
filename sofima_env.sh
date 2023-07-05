#!/bin/bash
conda create --name py311 -c conda-forge python=3.11 -y
conda activate py311
pip install git+https://github.com/google-research/sofima
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorstore