import functools as ft
import gc
import jax
import numpy as np
import tensorstore as ts

from sofima import flow_field

from eval_reg import utils, metrics


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
    
    # if sample_left: 
    #     mx = mx // 2
    # if sample_left:
    #     mx = mx + (mx // 2)
    # if sample_left: 
    #     mz = mz + (mz // 2)

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
                           tile_volumes: list[ts.TensorStore]
                           ) -> tuple[np.ndarray, np.ndarray]:
    layout_y, layout_x = tile_layout.shape

    # Output Containers, sofima uses cartesian convention
    conn_x = np.full((3, 1, layout_y, layout_x), np.nan)
    conn_y = np.full((3, 1, layout_y, layout_x), np.nan)

    # Row Pairs
    for y in range(layout_y): 
        for x in range(layout_x - 1):  # Stop one before the end 
            left_id = tile_layout[y, x]
            right_id = tile_layout[y, x + 1]
            left_tile = tile_volumes[left_id]
            right_tile = tile_volumes[right_id]

            conn_x[:, 0, y, x] = _estimate_h_offset_zyx(left_tile, right_tile)
            gc.collect()

            print(f'Left Id: {left_id}, Right Id: {right_id}')
            print(f'Left: ({y}, {x}), Right: ({y}, {x + 1})', conn_x[:, 0, y, x])

    # Column Pairs -- Reversed Loops
    for x in range(layout_x):
        for y in range(layout_y - 1):
            top_id = tile_layout[y, x]
            bot_id = tile_layout[y + 1, x]
            top_tile = tile_volumes[top_id]
            bot_tile = tile_volumes[bot_id]

            conn_y[:, 0, y, x] = _estimate_v_offset_zyx(top_tile, bot_tile)
            gc.collect()
            
            print(f'Top Id: {top_id}, Bottom Id: {bot_id}')
            print(f'Top: ({y}, {x}), Bot: ({y + 1}, {x})', conn_y[:, 0, y, x])

    return conn_x, conn_y

# New coarse registration window action: 
# Developing new functions so I do not break the ones above

def _r_estimate_h_offset_zyx(left_tile: ts.TensorStore, 
                             right_tile: ts.TensorStore, 
                             percent_overlap: float,
                             ) -> tuple[list[float], float]:
    """
    Args: 
    - left_tile: Search space
    - right_tile: Query space
    - percent_overlap: Percentage of tile size that is shared
        between neighboring tiles. Value between [0, 1).
    """
    
    tx, ty, tz = left_tile.shape
    mz = tz // 2
    overlap_offset = tx * percent_overlap
    overlap_width = tx * (1 - percent_overlap)

    # Huerisitc: Search patch shall be 1/2 width of the overlap region
    # Hueristic: Query patch shall be 3/4 size of search patch
    sp_width = 0.5 * (overlap_width)
    qp_width = 0.75 * sp_width

    # Defining centerpts of search/query patches
    # Hueristic: Search patch confined to overlap region bounds
    offsets = []
    for cx in np.linspace((0.5 * sp_width), 
                         overlap_width - (0.5 * sp_width), 
                         num=3):
        for cy in np.linspace(0.5 * sp_width, 
                             ty - (0.5 * sp_width),
                             num=3):
            cx = int(cx)
            cy = int(cy)
            
            # Search Tile
            search_cx = overlap_offset+cx
            left = left_tile[search_cx-(0.5 * sp_width):search_cx+(0.5 * sp_width),
                             cy-(0.5 * sp_width):cy+(0.5 * sp_width),
                             mz-(0.5 * sp_width):mz+(0.5 * sp_width)].read().result().T
           
            # Query Tile
            right = right_tile[cx-(0.5 * qp_width):cx+(0.5 * qp_width),
                               cy-(0.5 * qp_width):cy+(0.5 * qp_width),
                               mz-(0.5 * qp_width):mz+(0.5 * qp_width)].read().result().T

            # Produce Offset
            start_zyx = np.array(left.shape) // 2 - np.array(right.shape) // 2
            pc_init_zyx = np.array([0, 0, search_cx + start_zyx[2]])
            pc_zyx = np.array(_estimate_relative_offset_zyx(left, right))
            offsets.append(pc_init_zyx + pc_zyx)

    # Evaluate Offsets
    
    # If metric < threshold, return the inital tile position offset.



    # return best_offset  # offset with max metric. Worth outputing and seeing all though. 


def _r_estimate_v_offset_zyx(top_tile: ts.TensorStore, 
                           bot_tile: ts.TensorStore,
                           percent_overlap: float, 
                          ) -> tuple[list[float], float]:
    tx, ty, tz = top_tile.shape
    mz = tz // 2
    overlap_offset = ty * percent_overlap
    overlap_width = ty * (1 - percent_overlap)

    # Huerisitc: Search patch shall be 1/2 width of the overlap region
    # Hueristic: Query patch shall be 3/4 size of search patch
    sp_width = 0.5 * (overlap_width)
    qp_width = 0.75 * sp_width

    # Defining centerpts of search/query patches
    # Hueristic: Search patch confined to overlap region bounds
    offsets = []
    for cy in np.linspace((0.5 * sp_width), 
                         overlap_width - (0.5 * sp_width), 
                         num=3):
        for cx in np.linspace(0.5 * sp_width, 
                             tx - (0.5 * sp_width),
                             num=3):
            cy = int(cy)
            cx = int(cx)

            # Search Tile
            search_cy = overlap_offset+cy
            top = top_tile[search_cy-(0.5 * sp_width):search_cy+(0.5 * sp_width),
                           cy-(0.5 * sp_width):cy+(0.5 * sp_width),
                           mz-(0.5 * sp_width):mz+(0.5 * sp_width)].read().result().T
           
            # Query Tile
            bot = bot_tile[cx-(0.5 * qp_width):cx+(0.5 * qp_width),
                           cy-(0.5 * qp_width):cy+(0.5 * qp_width),
                           mz-(0.5 * qp_width):mz+(0.5 * qp_width)].read().result().T

            # Produce Offset
            start_zyx = np.array(top.shape) // 2 - np.array(bot.shape) // 2
            pc_init_zyx = np.array([0, search_cy + start_zyx[1], 0])    
            pc_zyx = np.array(_estimate_relative_offset_zyx(top, bot))
            offsets.append(pc_init_zyx + pc_zyx)

    # Evaluate Offsets
    

    # If metric < threshold, return the inital tile position offset.


    # return best_offset


# New global variables: 
METRIC = 'ISSM'
WINDOW_SIZE = 5
NUM_SAMPLES = 100

# def evaluate_offset(image_1: ts.TensorStore, 
#                     image_2: ts.TensorStore,
#                     transform: ):


#     bounds_1, bounds_2 = utils.calculate_bounds(
#             image_1_shape, image_2_shape, transform
#         )

#     # #Sample points in overlapping bounds
#     points = utils.sample_points_in_overlap(
#         bounds_1=bounds_1,
#         bounds_2=bounds_2,
#         numpoints=self.args["sampling_info"]["numpoints"],
#         sample_type=self.args["sampling_info"]["sampling_type"],
#         image_shape=image_1_shape,
#     )

#     # print("Points: ", points)

#     # Points that fit in window based on a window size
#     pruned_points = utils.prune_points_to_fit_window(
#         image_1_shape, points, self.args["window_size"]
#     )

#     discarded_points_window = points.shape[0] - pruned_points.shape[0]
#     LOGGER.info(
#         f"""Number of discarded points when prunning
#         points to window: {discarded_points_window}""",
#     )

#     # calculate metrics per images
#     metric_per_point = []

#     metric_calculator = ImageMetricsFactory().create(
#         image_1_data,
#         image_2_data,
#         self.args["metric"],
#         self.args["window_size"],
#     )

#     selected_pruned_points = []

#     for pruned_point in pruned_points:

#         met = metric_calculator.calculate_metrics(
#             point=pruned_point, transform=transform
#         )

#         if met:
#             selected_pruned_points.append(pruned_point)
#             metric_per_point.append(met)

#     # compute statistics
#     metric = self.args["metric"]
#     computed_points = len(metric_per_point)

#     dscrd_pts = points.shape[0] - discarded_points_window - computed_points
#     message = f"""Computed metric: {metric}
#     \nMean: {np.mean(metric_per_point)}
#     \nStd: {np.std(metric_per_point)}
#     """




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