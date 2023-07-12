import logging
import time

import dask
from dask.distributed import Client, LocalCluster
from distributed import wait
import xarray_multiscale
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
from ome_zarr.format import CurrentFormat
import zarr

from aind_data_transfer.transformations import ome_zarr
from aind_data_transfer.util import chunk_utils

import zarr_io

import dask.array as da
import numpy as np

def main(): 
  logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
  LOGGER = logging.getLogger(__name__)
  LOGGER.setLevel(logging.INFO)

  # Primary Input
  image = zarr_io.open_zarr_gcs('sofima-test-bucket', 'output_level_0_debug.zarr')
  class SyncAdapter:
    """Makes it possible to use a TensorStore objects as a numpy array."""
    
    def __init__(self, tstore):
      self.tstore = tstore

    def __getitem__(self, ind):
      return np.array(self.tstore[ind])

    def __getattr__(self, attr):
      return getattr(self.tstore, attr)

    @property
    def shape(self):
      return self.tstore.shape

    @property
    def ndim(self):
      return self.tstore.ndim

    @property
    def dtype(self):
      return "uint16"

  # Output Paths
  # output_path = 'gs://sofima-test-bucket/fused_multiscale.zarr/'
  # output_name = 'fused.zarr'
  output_path = "./fused_multiscale.zarr"

  # Other Input Parameters
  scale_factor = 2
  voxel_sizes = (0.176, 0.298, 0.298)

  # Actual Processing
  client = Client(LocalCluster(processes=True, threads_per_worker=1))

  dask_image = da.from_array(SyncAdapter(image))
  # This is the optimized chunksize
  chunks = chunk_utils.expand_chunks(chunks=dask_image.chunksize,
                                    data_shape=dask_image.shape,
                                    target_size=64, # Same as Cameron's code 
                                    itemsize=dask_image.itemsize) 
  chunks = chunk_utils.ensure_shape_5d(chunks)

  scale_axis = (1, 1, scale_factor, scale_factor, scale_factor)
  n_lvls = 5
  pyramid = xarray_multiscale.multiscale(
              dask_image,
              xarray_multiscale.reducers.windowed_mean,
              scale_axis,  # scale factors
              preserve_dtype=True,
              chunks="preserve",  # can also try "preserve", which is the default
              )[:n_lvls]

  pyramid_data = [arr.data for arr in pyramid]
  print(f'{pyramid_data=}')

  axes_5d = ome_zarr._get_axes_5d()
  transforms, chunk_opts = ome_zarr._compute_scales(
          len(pyramid),
          (scale_factor,) * 3,
          voxel_sizes,
          chunks, # Can optimize, or simply use dask default chunking. 
          pyramid[0].shape  # origin optional-- especially for single fused image
      )

  loader = CurrentFormat()
  # store = loader.init_store(output_path, mode='w')
  store = zarr.open(output_path, mode='w')
  print(store)

  # Actual Jobs
  LOGGER.info("Starting write...")
  t0 = time.time()
  jobs = write_multiscale(
      pyramid_data,
      group=store,
      fmt=CurrentFormat(),
      axes=axes_5d,
      coordinate_transformations=transforms,
      storage_options=chunk_opts,
      name=None,
      compute=False,
  )
  if jobs:
      LOGGER.info("Computing dask arrays...")
      arrs = dask.persist(*jobs)
      wait(arrs)
  write_time = time.time() - t0

  LOGGER.info(
        f"Finished writing fused image.\n"
        f"Took {write_time}s."
    )

if __name__ == '__main__':
  main()