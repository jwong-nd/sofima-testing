import functools as ft
import gc
import jax
import numpy as np
import tensorstore as ts

from sofima import flow_field


# Global/Data-Defined Variables for defining bounds of 'Search' and 'Query' volumes for cross correlation. 
QUERY_R_ORTHO = 25
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 25

SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 50


@ft.partial(jax.jit)
def _estimate_relative_offset_zyx(base, kernel
                                  ) -> list[float, float, float]:
    # Calculate FFT: left = base, right = kernel
    xc = flow_field.masked_xcorr(base, kernel, use_jax=True, dim=3)
    xc = xc.astype(np.float32)
    xc = xc[None, ...]

    # Find strongest peak in FFT, pass in FFT image center
    r = flow_field._batched_peaks(xc, ((xc.shape[1] + 1) // 2, (xc.shape[2] + 1) // 2, xc.shape[3] // 2), min_distance=2, threshold_rel=0.5)
    
    # r returns a list, relative offset is here
    relative_offset_xyz = r[0][0:3]
    return [relative_offset_xyz[2], relative_offset_xyz[1], relative_offset_xyz[0]]


def _estimate_h_offset_zyx(left_tile: ts.TensorStore, right_tile: ts.TensorStore
                           ) -> tuple[list[float], float]:
    tile_size_xyz = left_tile.shape
    mz = tile_size_xyz[2] // 2
    my = tile_size_xyz[1] // 2

    # Search Space, fixed
    left = left_tile[tile_size_xyz[0]-SEARCH_OVERLAP:,
                     my-SEARCH_R_ORTHO:my+SEARCH_R_ORTHO,
                     mz-SEARCH_R_ORTHO:mz+SEARCH_R_ORTHO].read().result().T
    
    # Query Patch, scanned against search space
    right = right_tile[QUERY_OVERLAP_OFFSET:QUERY_OVERLAP_OFFSET + QUERY_R_OVERLAP*2,
                       my-QUERY_R_ORTHO:my+QUERY_R_ORTHO,
                       mz-QUERY_R_ORTHO:mz+QUERY_R_ORTHO].read().result().T

    start_zyx = np.array(left.shape) // 2 - np.array(right.shape) // 2
    pc_init_zyx = np.array([tile_size_xyz[2] - SEARCH_OVERLAP + start_zyx[2], 0, 0])
    pc_zyx = np.array(_estimate_relative_offset_zyx(left, right))

    return pc_init_zyx + pc_zyx


def _estimate_v_offset_zyx(top_tile: ts.TensorStore, bot_tile: ts.TensorStore
                          ) -> tuple[list[float], float]:
    tile_size_xyz = top_tile.shape
    mz = tile_size_xyz[2] // 2
    mx = tile_size_xyz[0] // 2
    
    top = top_tile[mx-SEARCH_R_ORTHO:mx+SEARCH_R_ORTHO, 
                   tile_size_xyz[1]-SEARCH_OVERLAP:, 
                   mz-SEARCH_R_ORTHO:mz+SEARCH_R_ORTHO].read().result().T  
    bot = bot_tile[mx-QUERY_R_ORTHO:mx+QUERY_R_ORTHO, 
                   0:QUERY_R_OVERLAP*2, 
                   mz-QUERY_R_ORTHO:mz+QUERY_R_ORTHO].read().result().T

    start_zyx = np.array(top.shape) // 2 - np.array(bot.shape) // 2
    pc_init_zyx = np.array([0, tile_size_xyz[1] - SEARCH_OVERLAP + start_zyx[1], 0])
    pc_zyx = np.array(_estimate_relative_offset_zyx(top, bot))

    # print(top.shape)
    # print(bot.shape)
    # print(start_zyx)
    # print(pc_init_zyx)
    # print(pc_zyx)

    return pc_init_zyx + pc_zyx


def compute_coarse_offsets(yx_shape: tuple[int, int],
                           tile_map: dict[tuple[int, int], int], 
                           tile_volumes: list[ts.TensorStore]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Computes coarse offsets for every tile.

    Args:
    yx_shape: shape of the tile grid
    tile_map: maps YX tile coordinates to tile IDs
    """

    # Output Containers
    conn_x = np.full((3, 1, yx_shape[0], yx_shape[1]), np.nan)
    conn_y = np.full((3, 1, yx_shape[0], yx_shape[1]), np.nan)

    # Fill conn_x
    for x in range(0, yx_shape[1] - 1):
        for y in range(0, yx_shape[0]):
            if not ((y, x) in tile_map and (y, x + 1) in tile_map):
                continue

            left = tile_map[(y, x)]
            left_tile = tile_volumes[left]
            right = tile_map[(y, x + 1)]
            right_tile = tile_volumes[right]

            conn_x[:, 0, y, x] = _estimate_h_offset_zyx(left_tile, right_tile)
            gc.collect() 

            # print(f'Left: ({x}, {y}), Right: ({x + 1}, {y})', conn_x[:, 0, y, x])

    # Fill conn_y
    for y in range(0, yx_shape[0] - 1):
        for x in range(0, yx_shape[1]):
            if not ((y, x) in tile_map and (y + 1, x) in tile_map):
                continue

            top = tile_map[(y, x)]
            top_tile = tile_volumes[top]
            bot = tile_map[(y + 1, x)]
            bot_tile = tile_volumes[bot]
            conn_y[:, 0, y, x] = _estimate_v_offset_zyx(top_tile, bot_tile)
            gc.collect() 

            # print(f'Top: ({x}, {y}), Bot: ({x}, {y + 1})', conn_y[:, 0, y, x])

    return conn_x, conn_y