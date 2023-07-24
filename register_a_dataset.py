import logging
from pathlib import Path
import time
import yaml

import ng_utils
from sofima.zarr import zarr_io, zarr_register_and_fuse_3d

def main(): 
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    # Load Configuration File 
    with open('config.yml', 'r') as file:
        params = yaml.safe_load(file)

    cloud_storage = zarr_io.CloudStorage.S3 
    bucket = params['input']['bucket']
    dataset_path = params['input']['dataset_path']
    channel = params['input']['registration_channel']
    downsample_exp = params['input']['downsample_exp']

    output_cloud_storage = zarr_io.CloudStorage.GCS
    output_bucket = params['output']['bucket']
    output_path = params['output']['name']
    output_downsample_exp = params['output']['downsample_exp']

    # Create Dataset
    zd = zarr_io.DiSpimDataset(cloud_storage, 
                            bucket,
                            dataset_path,
                            channel,
                            downsample_exp)

    # Run Coarse Registration
    zarr_stitcher = zarr_register_and_fuse_3d.ZarrStitcher(zd)
    cx, cy, coarse_mesh = zarr_stitcher.run_coarse_registration()

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
    save_mesh_path = f"{Path(dataset_path).root}_mesh.npy"
    zarr_stitcher.run_fine_registration(cx, 
                                        cy,
                                        coarse_mesh,
                                        stride_zyx=(20, 20, 20), 
                                        save_mesh_path=save_mesh_path)

    # Run Fusion
    t0 = time.time()
    zarr_stitcher.run_fusion(output_cloud_storage=output_cloud_storage,
                            output_bucket=output_bucket,
                            output_path=output_path,
                            downsample_exp=output_downsample_exp,
                            cx=cx,
                            cy=cy,
                            tile_mesh_path=save_mesh_path)
    fusion_time = time.time() - t0

    LOGGER.info(
        f"Fusion: {fusion_time}"
    )

if __name__ == '__main__': 
    main()