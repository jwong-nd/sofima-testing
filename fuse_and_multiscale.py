import logging
import numpy as np
import time

from config import PipelineConfiguration
import cloud_utils
from sofima.zarr import zarr_io, zarr_register_and_fuse_3d
from multiscale_blocked_writer import run_multiscale


# Current hardware: 64 CPU's, 256 GB
def main(): 
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    pc = PipelineConfiguration()
    cloud_storage = zarr_io.CloudStorage.S3 
    bucket = pc.params['input_bucket']
    dataset_path = pc.params['input_dataset_path']
    registration_channel = pc.params['registration_channel']
    downsample_exp = pc.params['downsample_exp']

    cloud_utils.read_from_bucket_gcs(pc.params['home_bucket'],
                                     f'SOFIMA_{pc.dataset_name}/{pc.COARSE_MESH_NAME}',
                                     pc.COARSE_MESH_NAME)
    data = np.load(pc.COARSE_MESH_NAME)
    cx = data['cx']
    cy = data['cy']
    coarse_mesh = data['coarse_mesh']

    cloud_utils.read_from_bucket_gcs(pc.params['home_bucket'],
                                     f'SOFIMA_{pc.dataset_name}/{pc.ELASTIC_MESH_NAME}',
                                     pc.ELASTIC_MESH_NAME)

    zd = zarr_io.DiSpimDataset(cloud_storage, 
                            bucket,
                            dataset_path,
                            registration_channel,
                            downsample_exp)
    zarr_stitcher = zarr_register_and_fuse_3d.ZarrStitcher(zd)
    LOGGER.info(f'Fusing Dataset: {pc.dataset_name}')

    # Run fusion
    t0 = time.time()
    zarr_stitcher.run_fusion(output_cloud_storage=zarr_io.CloudStorage.GCS,
                            output_bucket=pc.params['home_bucket'],
                            output_path=f'SOFIMA_{pc.dataset_name}/{pc.FULL_RES_NAME}',
                            downsample_exp=0,
                            cx=cx,
                            cy=cy,
                            tile_mesh_path=pc.ELASTIC_MESH_NAME, 
                            parallelism=48) # Increased! 
    fusion_time = time.time() - t0
    LOGGER.info(f'Fusion Time: {fusion_time}')

    # Run multiscale
    t0 = time.time()
    run_multiscale(pc.params['home_bucket'],
                   f'SOFIMA_{pc.dataset_name}/{pc.FULL_RES_NAME}', 
                   pc.params['home_bucket'], 
                   f'SOFIMA_{pc.dataset_name}/{pc.FUSION_NAME}',
                   zd.vox_size_xyz[::-1])
    multiscale_time = time.time() - t0
    LOGGER.info(f'Multiscale Time: {multiscale_time}')


if __name__ == '__main__':
    main()