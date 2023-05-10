from concurrent import futures
import numpy as np
import skimage.exposure 
import tensorstore as ts

import zarr_io

MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 800

# Runs CLAHE normalization and converts data to uint8.
# def preprocess_tiles(tile_volumes: list[ts.TensorStore], output_names: list[str], bucket: str):
#     with futures.ThreadPoolExecutor() as tpe:
#         for src_vol, output_name in zip(tile_volumes, output_names):
#             out_vol = zarr_io.write_zarr(bucket, src_vol.shape, output_name)
#             fs = set()
#             for z in range(out_vol.shape[-1]):
#                 def _clahe(z):
#                     sec = src_vol[:, :, z].read().result()
#                     sec = np.clip(sec, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
#                     clahed = skimage.exposure.equalize_adapthist(sec, clip_limit=0.03)
#                     out_vol[:, :, z].write((clahed * 255).astype(np.uint8)).result()
#                 fs.add(tpe.submit(_clahe, z))

#             for f in futures.as_completed(fs):
#                 f.result()


def preprocess_tiles(tile_volumes: list[ts.TensorStore], output_names: list[str], bucket: str):
    with futures.ThreadPoolExecutor() as tpe: 
        for src_vol, output_name in zip(tile_volumes, output_names):
            out_vol = zarr_io.write_zarr(bucket, src_vol.shape, output_name)
            fs = set()
            for z in range(out_vol.shape[2]):  # Iterate through z planes
                def _clahe(z):
                    sec = src_vol[0, 0, z, :, :].read().result()
                    sec = np.clip(sec, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
                    clahed = skimage.exposure.equalize_adapthist(sec, clip_limit=0.03)
                    out_vol[0, 0, z, :, :].write((clahed * 255).astype(np.uint8)).result()
                fs.add(tpe.submit(_clahe, z))
            
            for f in futures.as_completed(fs):
                f.result()