import logging
import numpy as np
from pathlib import Path
import time

from config import PipelineConfiguration
import cloud_utils
import ng_utils
from sofima.zarr import zarr_io, zarr_register_and_fuse_3d


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

    # Create Dataset
    zd = zarr_io.DiSpimDataset(cloud_storage, 
                            bucket,
                            dataset_path,
                            registration_channel,
                            downsample_exp)
    LOGGER.info(f'Registering Dataset: {pc.dataset_name}')

    # Run Coarse Registration
    t0 = time.time()
    zarr_stitcher = zarr_register_and_fuse_3d.ZarrStitcher(zd)
    cx, cy, coarse_mesh = zarr_stitcher.run_coarse_registration()
    np.savez_compressed(pc.COARSE_MESH_NAME,
                        cx=cx, 
                        cy=cy,
                        coarse_mesh=coarse_mesh)
    cloud_utils.write_to_bucket_gcs(pc.params['home_bucket'],
                                    f'SOFIMA_{pc.dataset_name}/{pc.COARSE_MESH_NAME}',
                                    pc.COARSE_MESH_NAME)
    coarse_reg_time = time.time() - t0
    LOGGER.info(f'Coarse Registration Time: {coarse_reg_time}')

    # Verify Coarse Registration Looks Good
    zd_list = [zd]
    remaining_channels = list(set(zd.channels) - set([channel]))
    for channel in remaining_channels: 
        zd_list.append(zarr_io.DiSpimDataset(cloud_storage, 
                                            bucket, 
                                            dataset_path, 
                                            channel, 
                                            downsample_exp))
    ng_utils.ng_link_multi_channel(zd_list, coarse_mesh)
    
    # Run Elastic Registration
    t0 = time.time()
    zarr_stitcher.run_fine_registration(cx, 
                                        cy,
                                        coarse_mesh,
                                        stride_zyx=(20, 20, 20), 
                                        save_mesh_path=pc.ELASTIC_MESH_NAME)
    cloud_utils.write_to_bucket_gcs(pc.params['home_bucket'],
                                    f'SOFIMA_{pc.dataset_name}/{pc.ELASTIC_MESH_NAME}',
                                    pc.ELASTIC_MESH_NAME)
    elastic_reg_time = time.time() - t0
    LOGGER.info(f'Elastic Registration Time: {elastic_reg_time}')

if __name__ == '__main__': 
    main()