import logging
import time
import dask
# from dask.distributed import Client, LocalCluster
import numpy as np
import zarr
from ome_zarr.io import parse_url

from aind_data_transfer.util.io_utils import BlockedArrayWriter
from aind_data_transfer.util.chunk_utils import ensure_shape_5d, ensure_array_5d
from aind_data_transfer.transformations.ome_zarr import (
    store_array,
    downsample_and_store,
    _get_bytes,
    write_ome_ngff_metadata
)

import zarr_io

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

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

def run_multiscale(input_bucket: str, 
                   input_name: str, 
                   output_gcs_bucket: str,  
                   output_name: str, 
                   voxel_sizes_zyx: tuple): 
    
    image = zarr_io.open_zarr_gcs(input_bucket, input_name)
    arr = dask.array.from_array(SyncAdapter(image))
    arr = arr.rechunk((1, 1, 128, 128, 128))
    arr = ensure_array_5d(arr)
    LOGGER.info(f"input array: {arr}")
    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    
    output_path = f"gs://{output_gcs_bucket}/{output_name}"
    group = zarr.open_group(output_path, mode='w')

    scale_factors = (2, 2, 2) 
    scale_factors = ensure_shape_5d(scale_factors)

    n_levels = 5
    compressor = None

    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")

    # Actual Processing
    write_ome_ngff_metadata(
            group,
            arr,
            output_name,
            n_levels,
            scale_factors[2:],
            voxel_sizes_zyx,
            origin=None,
        )

    t0 = time.time()
    store_array(arr, group, "0", block_shape, compressor)
    pyramid = downsample_and_store(
        arr, group, n_levels, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )

def main(): 
    # Format Input Zarr -> Dask Array
    input_bucket = 'sofima-test-bucket' 
    # input_name = 'fused_diSPIM_647403_2022-11-21_23-12-18_CH_405_CAM_1'
    # input_name = 'fused_diSPIM_647459_2022-12-07_00-00-00_CH_405_CAM_0'
    input_name = 'output_level_0_debug.zarr'

    image = zarr_io.open_zarr_gcs(input_bucket, input_name)
    arr = dask.array.from_array(SyncAdapter(image))
    arr = arr.rechunk((1, 1, 128, 128, 128))
    arr = ensure_array_5d(arr)
    LOGGER.info(f"input array: {arr}")
    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")

    # Other Input Parameters
    # output_path = "./fused_multiscale.zarr"
    # group = zarr.open(output_path, mode='w')
    # image_name = 'fused_multiscale_diSPIM_647403_2022-11-21_23-12-18_CH_405_CAM_1.zarr'
    # image_name = 'fused_multiscale_diSPIM_647459_2022-12-07_00-00-00_CH_405_CAM_0'
    image_name = 'fused_multiscale_output_level_0.zarr'

    output_path = f"gs://sofima-test-bucket/{image_name}"
    group = zarr.open_group(output_path, mode='w')

    scale_factors = (2, 2, 2) 
    scale_factors = ensure_shape_5d(scale_factors)

    n_levels = 5
    compressor = None
    voxel_sizes = (0.176, 0.298, 0.298)

    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")

    # Actual Processing
    write_ome_ngff_metadata(
            group,
            arr,
            image_name,
            n_levels,
            scale_factors[2:],
            voxel_sizes,
            origin=None,
        )

    t0 = time.time()
    store_array(arr, group, "0", block_shape, compressor)
    pyramid = downsample_and_store(
        arr, group, n_levels, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )

if __name__ == '__main__':
    # run_multiscale(input_bucket='sofima-test-bucket', 
    #                input_name='fused_image_name.zarr',
    #                output_gcs_bucket='sofima-test-bucket', 
    #                output_name='multiscale_fused_X.zarr', 
    #                voxel_sizes_zyx=(0, 0, 0))  # Inside zd

    main()