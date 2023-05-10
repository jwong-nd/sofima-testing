import gc 
import functools as ft
import numpy as np
import jax
# import jax.numpy as jnp

from sofima import flow_field
# from sofima import mesh
# from sofima import stitch_rigid

import tensorstore as ts


# Global/Data-Defined Variables for defining bounds of 'Search' and 'Query' volumes for cross correlation. 
# Original Variables
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 100
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 400
SEARCH_R_ORTHO = 600

# New Variables for 2-Downsampled Dataset, shape: (1, 1, 3551, 576, 576)
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 0
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 200
SEARCH_R_ORTHO = 200

# Updated variables, expanding search region: 
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 250  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 250

# Updated variables, deepen search region: 
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 400  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 250

# Updated variables, deepen search region: 
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 500  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 250

# Updated variables, deepen query region: 
# QUERY_R_ORTHO = 50
# QUERY_OVERLAP_OFFSET = 100  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 50

# SEARCH_OVERLAP = 500  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 250


# Added from notebook:
# tile_size_xyz = (1152, 1152, 7103)
tile_size_xyz = (576, 576, 3551)

@ft.partial(jax.jit, static_argnames=('overlap',))
def _estimate_horiz(left, right, overlap):
    xc = flow_field.masked_xcorr(left, right, use_jax=True, dim=3)
    xc = xc.astype(np.float32)
    xc = xc[None, ...]
    r = flow_field._batched_peaks(xc, ((xc.shape[1] + 1) // 2, (xc.shape[2] + 1) // 2, overlap // 2), min_distance=2, threshold_rel=0.5)
    return r[0]

@ft.partial(jax.jit, static_argnames=('overlap',))
def _estimate_vert(top, bot, overlap):
    xc = flow_field.masked_xcorr(top, bot, use_jax=True, dim=3)
    xc = xc.astype(np.float32)
    xc = xc[None, ...]
    r = flow_field._batched_peaks(xc, ((xc.shape[1] + 1) // 2, overlap // 2, (xc.shape[3] + 1) // 2), min_distance=2, threshold_rel=0.5)
    return r[0]

def _estimate_offset_horiz(overlap: int, left: int, right: int, tile_volumes: list[ts.TensorStore]) -> tuple[list[float], float]:
    mz = tile_size_xyz[2] // 2
    my = tile_size_xyz[1] // 2

    left = tile_volumes[left][tile_size_xyz[0]-overlap:,
                            my-SEARCH_R_ORTHO:my+SEARCH_R_ORTHO,
                            mz-SEARCH_R_ORTHO:mz+SEARCH_R_ORTHO].read().result().T
    right = tile_volumes[right][QUERY_OVERLAP_OFFSET:QUERY_OVERLAP_OFFSET + QUERY_R_OVERLAP*2,
                              my-QUERY_R_ORTHO:my+QUERY_R_ORTHO,
                              mz-QUERY_R_ORTHO:mz+QUERY_R_ORTHO].read().result().T

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left = left - left.mean()
    right = right - right.mean()

    r = np.array(_estimate_horiz(left, right, overlap))

    # the section at: overlap // 2 + r[0] relative to the start of left corresponds to right[100]
    # so the true overlap between the two blocks is:
    # 100 + overlap - ((overlap // 2) + r[0])

    return [QUERY_OVERLAP_OFFSET + QUERY_R_OVERLAP + overlap - ((overlap // 2) + r[0]), r[1], r[2]], abs(r[4])

def _estimate_offset_vert(overlap: int, top: int, bot: int, tile_volumes: list[ts.TensorStore]) -> tuple[list[float], float]:
    mz = tile_size_xyz[2] // 2
    mx = tile_size_xyz[0] // 2

    top_x_lb = mx - SEARCH_R_ORTHO
    top_x_ub = mx + SEARCH_R_ORTHO
    top_y_lb = tile_size_xyz[1] - overlap
    top_y_ub = tile_size_xyz[1]
    top_z_lb = mz - SEARCH_R_ORTHO
    top_z_ub = mz + SEARCH_R_ORTHO
    print(f'{top_x_lb=}')
    print(f'{top_x_ub=}')
    print(f'{top_y_lb=}')
    print(f'{top_y_ub=}')
    print(f'{top_z_lb=}')
    print(f'{top_z_ub=}')

    bot_x_lb = mx - QUERY_R_ORTHO
    bot_x_ub = mx + QUERY_R_ORTHO
    bot_y_lb = QUERY_OVERLAP_OFFSET
    bot_y_ub = QUERY_OVERLAP_OFFSET + QUERY_R_OVERLAP * 2
    bot_z_lb = mz - QUERY_R_ORTHO
    bot_z_ub = mz + QUERY_R_ORTHO
    print(f'{bot_x_lb=}')
    print(f'{bot_x_ub=}')
    print(f'{bot_y_lb=}')
    print(f'{bot_y_ub=}')
    print(f'{bot_z_lb=}')
    print(f'{bot_z_ub=}')
    
    top = tile_volumes[top][top_x_lb:top_x_ub, top_y_lb:, top_z_lb:top_z_ub].read().result().T  
    print('Top shape', top.shape)
        
    bot = tile_volumes[bot][bot_x_lb:bot_x_ub, bot_y_lb:bot_y_ub, bot_z_lb:bot_z_ub].read().result().T
    print('Bot shape', bot.shape)    
    
    top = top.astype(np.float32)
    bot = bot.astype(np.float32)
    top = top - top.mean()
    bot = bot - bot.mean()
    r = np.array(_estimate_vert(top, bot, overlap))
    print('r, result of cross correlation', r)
    
    return [r[0], QUERY_OVERLAP_OFFSET + QUERY_R_OVERLAP + overlap - ((overlap // 2) + r[1]), r[2]], abs(r[4])
# ^Don't understand why cannot simply return r[0]. Anyway, let's try to move on. 


# Added 'tile_volumes' to pass onto {_estimate_offset_horiz(), _estimate_offset_vert()}
def compute_coarse_offsets(yx_shape: tuple[int, int],
                           tile_map: dict[tuple[int, int], int], 
                           tile_volumes: list[ts.TensorStore]) -> tuple[np.ndarray, np.ndarray]:
    """Computes coarse offsets for every tile.

    Args:
    yx_shape: shape of the tile grid
    tile_map: maps YX tile coordinates to tile IDs
    """

    def _find_offset(estimate_fn, pre: int, post: int, axis: int):
        offset, pr = estimate_fn(SEARCH_OVERLAP, pre, post, tile_volumes)
        # Transform the number of overlapping voxels into an offset vector that
        # can be applied to the 'post' tile.
        offset[axis] = - offset[axis]
        return offset

    conn_x = np.full((3, 1, yx_shape[0], yx_shape[1]), np.nan)  # Init a 3D vector even tho we only fill out two entries
    for x in range(0, yx_shape[1] - 1):
        for y in range(0, yx_shape[0]):
            if not ((y, x) in tile_map and (y, x + 1) in tile_map):
                continue
            print('processing', x, y)

            left = tile_map[(y, x)]
            right = tile_map[(y, x + 1)]
            gc.collect()
            conn_x[:, 0, y, x] = _find_offset(_estimate_offset_horiz, left, right, 0)
            # print(conn_x[:, 0, y, x])

    conn_y = np.full((3, 1, yx_shape[0], yx_shape[1]), np.nan)
    for y in range(0, yx_shape[0] - 1):
        for x in range(0, yx_shape[1]):
            if not ((y, x) in tile_map and (y + 1, x) in tile_map):
                continue
            print('processing', x, y)

            top = tile_map[(y, x)]
            bot = tile_map[(y + 1, x)]
            gc.collect()

            conn_y[:, 0, y, x] = _find_offset(_estimate_offset_vert, top, bot, 1)
            print(conn_y[:, 0, y, x])

    return conn_x, conn_y

