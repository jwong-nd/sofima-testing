import functools as ft
import gc
import jax
import numpy as np
import tensorstore as ts

from sofima import flow_field


# Small Query Size (Dispim)
# QUERY_R_ORTHO = 25
# QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 25

# SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 50

# Normal Query Size (Dispim):
QUERY_R_ORTHO = 50
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 50

SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 50

# Large Query size (Dispim):
# QUERY_R_ORTHO = 100
# QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 100

# SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 100

# Custom Parameters for hole dataset:
# QUERY_R_ORTHO = 100
# QUERY_OVERLAP_OFFSET = 400  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 100

# SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 100

# Custom Parameters for wave dataset:
# QUERY_R_ORTHO = 120
# QUERY_OVERLAP_OFFSET = 400  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 120

# SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 100

# Exaspim Parameters: 
# QUERY_R_ORTHO = 100
# QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 100

# SEARCH_OVERLAP = 400  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 50

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
    pc_init_zyx = np.array([0, 0, tile_size_xyz[0] - SEARCH_OVERLAP + start_zyx[2]])
    pc_zyx = np.array(_estimate_relative_offset_zyx(left, right))

    return pc_init_zyx + pc_zyx

# Okay, let's use peak sharpness. 
# Its a ratio of the height to the base of the peak, where if base is ~equivalent for every peak 
# we are comparing apples to apples. Greater height is better, but need to define height relative to a base. 

def _estimate_v_offset_zyx(top_tile: ts.TensorStore, bot_tile: ts.TensorStore,
                           sample_left = False, 
                          ) -> tuple[list[float], float]:
    tile_size_xyz = top_tile.shape
    mz = tile_size_xyz[2] // 2
    mx = tile_size_xyz[0] // 2
    
    if sample_left: 
        mx = mx // 2
    # if sample_right:
    #     mx = mx + (mx // 2)

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


# Tweaking is a bit of a hassle-- 
# Intermediate step to auto-tweaking, define patches 
# based on image size.

def compute_coarse_offsets(tile_layout: np.ndarray, 
                           tile_volumes: list[ts.TensorStore],
                           sample_left = False) -> tuple[np.ndarray, np.ndarray]:
    # Using numpy axis convention
    layout_x, layout_y = tile_layout.shape

    # Output Containers, sofima uses cartesian convention
    conn_x = np.full((3, 1, layout_x, layout_y), np.nan)
    conn_y = np.full((3, 1, layout_x, layout_y), np.nan)

    # Row Pairs
    for x in range(layout_x): 
        for y in range(layout_y - 1):  # Stop one before the end 
            left_id = tile_layout[x, y]
            right_id = tile_layout[x, y + 1]
            left_tile = tile_volumes[left_id]
            right_tile = tile_volumes[right_id]

            conn_x[:, 0, x, y] = _estimate_h_offset_zyx(left_tile, right_tile)
            gc.collect()

            print(f'Left Id: {left_id}, Right Id: {right_id}')
            print(f'Left: ({x}, {y}), Right: ({x}, {y + 1})', conn_x[:, 0, x, y])

    # Column Pairs
    for y in range(layout_y):
        for x in range(layout_x - 1):
            top_id = tile_layout[x, y]
            bot_id = tile_layout[x + 1, y]
            top_tile = tile_volumes[top_id]
            bot_tile = tile_volumes[bot_id]

            conn_y[:, 0, x, y] = _estimate_v_offset_zyx(top_tile, bot_tile, sample_left)
            gc.collect()
            
            print(f'Top Id: {top_id}, Bottom Id: {bot_id}')
            print(f'Top: ({x}, {y}), Bot: ({x + 1}, {y})', conn_y[:, 0, x, y])

    return conn_x, conn_y