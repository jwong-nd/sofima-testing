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
# QUERY_R_ORTHO = 50
# QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
# QUERY_R_OVERLAP = 50

# SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
# SEARCH_R_ORTHO = 50

# Large Query size (Dispim):
QUERY_R_ORTHO = 100
QUERY_OVERLAP_OFFSET = 0  # Overlap = 'starting line' in neighboring tile
QUERY_R_OVERLAP = 100

SEARCH_OVERLAP = 300  # Boundary - overlap = 'starting line' in search tile
SEARCH_R_ORTHO = 100

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

# Horizontal / Vertical appear to follow a different
# convention. Fix that. 

# Will return to this, 
# Addressing Michal's comments

# # Updated return function!
# @ft.partial(jax.jit)
# def _estimate_relative_offset_zyx(base, kernel
#                                   ) -> list[float, float, float]:
#     # Calculate FFT: left = base, right = kernel
#     xc = flow_field.masked_xcorr(base, kernel, use_jax=True, dim=3)
#     xc = xc.astype(np.float32)
#     xc = xc[None, ...]

#     # Find strongest peak in FFT, pass in FFT image center
#     r = flow_field._batched_peaks(xc, ((xc.shape[1] + 1) // 2, (xc.shape[2] + 1) // 2, xc.shape[3] // 2), min_distance=2, threshold_rel=0.5)
    
#     # r returns a list, relative offset is here
#     relative_offset_xyz = r[0][0:3]
#     return [relative_offset_xyz[2], relative_offset_xyz[1], relative_offset_xyz[0]], r[0][4]


# # Search overlap defined in pixels. 
# def _estimate_h_offset_zyx(left_tile: ts.TensorStore, right_tile: ts.TensorStore,
#                            search_overlap: int) -> tuple[list[float], float]:
#     tile_size_xyz = left_tile.shape
#     x, y, z = tile_size_xyz
#     mz = z // 2

#     # Search Patch Size is 1/2 in each dimension
#     # Query Patch Size is 1/4 each dimension
#     # Sweeps across Left, Middle, and Right Patch-Pair Biases.
#     offset = None
#     max_peak_stat = 0
#     for bias in [y // 4, y // 2, 3 * (y // 4)]:
#         left = left_tile[x - search_overlap:, 
#                         bias - (y // 4): bias + (y // 4),
#                         mz - (z // 4): mz + (z // 4)].read().result().T 
#         right = right_tile[0:(x // 4),
#                         bias - (y // 8):bias + (y // 8),
#                         mz - (z // 8): mz + (z // 8)].read().result().T 

#         pc_zyx, peak_stat = _estimate_relative_offset_zyx(left, right)
        
#         if peak_stat > max_peak_stat:
#             start_zyx = np.array(left.shape) // 2 - np.array(right.shape) // 2
#             pc_init_zyx = np.array([0, 0, x - search_overlap + start_zyx[2]])    
#             offset = pc_init_zyx + np.array(pc_zyx)

#         max_peak_stat = max(max_peak_stat, peak_stat)

#     return offset


# # Search overlap defined in pixels. 
# def _estimate_v_offset_zyx(top_tile: ts.TensorStore, bot_tile: ts.TensorStore, 
#                           search_overlap: int) -> tuple[list[float], float]:
#     tile_size_xyz = top_tile.shape
#     x, y, z = tile_size_xyz
#     my = y - search_overlap
#     mz = z // 2
    
#     print(tile_size_xyz)

#     # Last try--
#     # Will try sweeping scale parameters
#     # Intuitively, we should search larger areas with different x/y biases. 

#     # Search Patch Size is 1/2 in each dimension
#     # Query Patch Size is 1/4 each dimension
#     # Sweeps across Left, Middle, and Right Patch-Pair Biases.
#     offset = None
#     max_peak_stat = 0
#     for x_bias in [x // 4, x // 2, 3 * (x // 4)]:
#         for y_bias in [0, y // 8, y // 4]:
#             top = top_tile[x_bias - (x // 8): x_bias + (x // 8),
#                             my - (y // 4): my + (y // 4), 
#                             mz - (y // 8): mz + (y // 8)].read().result().T 
#             bot = bot_tile[x_bias - (x // 8): x_bias + (x // 8),
#                             y_bias: y_bias + (y // 3),
#                             mz - (y // 8): mz + (y // 8)].read().result().T 

#             print(top.shape)
#             print(bot.shape)

#             pc_zyx, peak_stat = _estimate_relative_offset_zyx(top, bot)

#             print(f'{x_bias=}')
#             print(f'{y_bias=}')
#             print(f'{peak_stat=}')

#             start_zyx = np.array(top.shape) // 2 - np.array(bot.shape) // 2
#             pc_init_zyx = np.array([0, y - search_overlap + start_zyx[1], 0])    
#             print(f'current_offset={pc_init_zyx + np.array(pc_zyx)}')

#             if peak_stat > max_peak_stat:
#                 start_zyx = np.array(top.shape) // 2 - np.array(bot.shape) // 2
#                 pc_init_zyx = np.array([0, y - search_overlap + start_zyx[1], 0])    
#                 offset = pc_init_zyx + np.array(pc_zyx)

#             max_peak_stat = max(max_peak_stat, peak_stat)

#     return offset


# # Search overlap is in pixels
# def compute_coarse_offsets(tile_layout: np.ndarray, 
#                            tile_volumes: list[ts.TensorStore],
#                            search_overlap: int) -> tuple[np.ndarray, np.ndarray]:
#     # Using numpy axis convention
#     layout_x, layout_y = tile_layout.shape

#     # Output Containers, sofima uses cartesian convention
#     conn_x = np.full((3, 1, layout_x, layout_y), np.nan)
#     conn_y = np.full((3, 1, layout_x, layout_y), np.nan)

#     # Row Pairs
#     for x in range(layout_x): 
#         for y in range(layout_y - 1):  # Stop one before the end 
#             left_id = tile_layout[x, y]
#             right_id = tile_layout[x, y + 1]
#             left_tile = tile_volumes[left_id]
#             right_tile = tile_volumes[right_id]

#             conn_x[:, 0, x, y] = _estimate_h_offset_zyx(left_tile, right_tile, search_overlap)
#             gc.collect()

#             print(f'Left Id: {left_id}, Right Id: {right_id}')
#             print(f'Left: ({x}, {y}), Right: ({x}, {y + 1})', conn_x[:, 0, x, y])

#     # Column Pairs
#     for y in range(layout_y):
#         for x in range(layout_x - 1):
#             top_id = tile_layout[x, y]
#             bot_id = tile_layout[x + 1, y]
#             top_tile = tile_volumes[top_id]
#             bot_tile = tile_volumes[bot_id]

#             conn_y[:, 0, x, y] = _estimate_v_offset_zyx(top_tile, bot_tile, search_overlap)
#             gc.collect()
            
#             print(f'Top Id: {top_id}, Bottom Id: {bot_id}')
#             print(f'Top: ({x}, {y}), Bot: ({x + 1}, {y})', conn_y[:, 0, x, y])

#     return conn_x, conn_y