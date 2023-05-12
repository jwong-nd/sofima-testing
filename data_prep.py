import numpy as np
import tensorstore as ts

import kornia
import torch

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

# def preprocess_tiles(tile_volumes: list[ts.TensorStore], output_names: list[str], bucket: str):
#     with futures.ThreadPoolExecutor() as tpe: 
#         for src_vol, output_name in zip(tile_volumes, output_names):
#             out_vol = zarr_io.write_zarr(bucket, src_vol.shape, output_name)
#             fs = set()
#             for z in range(out_vol.shape[2]):  # Iterate through z planes
#                 def _clahe(z):
#                     sec = src_vol[0, 0, z, :, :].read().result()
#                     sec = np.clip(sec, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
#                     clahed = skimage.exposure.equalize_adapthist(sec, clip_limit=0.03)
#                     out_vol[0, 0, z, :, :].write((clahed * 255).astype(np.uint8)).result()
#                 fs.add(tpe.submit(_clahe, z))
            
#             for f in futures.as_completed(fs):
#                 f.result()

def preprocess_tiles(tile_volumes: list[ts.TensorStore], output_names: list[str], bucket: str, batch_size: int=32):
    BATCH_SIZE = batch_size
    for src_vol, output_name in zip(tile_volumes, output_names):  # Iterate through volumes
        out_vol = zarr_io.write_zarr(bucket, src_vol.shape, output_name)

        _, _, z, y, x = out_vol.shape
        num_batches = int(np.ceil(z / BATCH_SIZE))
        
        for i in range(num_batches - 1):  # Iterate through batches of planes
            batch_index = i * BATCH_SIZE
            sec = src_vol[0, 0, batch_index:batch_index + BATCH_SIZE + 1, :, :].read().result()
            sec = np.clip(sec, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
            
            # Formatting to (B, C, H, W): 
            # (1, z, y, x) -> (z, 1, y, x)
            sec_z, _, _ = sec.shape  
            sec = sec[np.newaxis, :]
            sec = sec.reshape(sec_z, 1, y, x)
            sec = np.float16(sec)
            sec = torch.tensor(sec, device='cuda')
            
            sec = kornia.enhance.normalize_min_max(sec)
            clahed = kornia.enhance.equalize_clahe(sec)  # Keeping default parameters

            # Formatting back to (z, y, x)
            clahed = sec.cpu().numpy()
            clahed = clahed.reshape(1, sec_z, y, x)
            clahed = clahed[0, :, :, :]

            clahed = clahed * 255
            clahed = np.uint8(clahed) 

            out_vol[0, 0, batch_index:batch_index + BATCH_SIZE + 1, :, :].write(clahed).result()