import numpy as np

from typing import Any, Mapping, Sequence, Union
from sofima import flow_field
from connectomics.common import bounding_box

# As opposed to .result() the entire tensorstore, 
# this splices the tensorstore, then .result() under numpy syntax.
class SyncAdapter:
  """Makes it possible to use a TensorStore objects as a numpy array."""
  def __init__(self, tstore, i):
    self.tstore = tstore
    self.i = i

  def __getitem__(self, ind):
    print(ind)
    return np.array(self.tstore[ind])

  def __getattr__(self, attr):
    return getattr(self.tstore, attr)

  @property
  def shape(self):
    return self.tstore.shape

  @property
  def ndim(self):
    return self.tstore.ndim


Vector = Union[tuple[int, int], tuple[int, int, int]]  # [z]yx order
TileXY = tuple[int, int]
ShapeXYZ = tuple[int, int, int]
TileFlow = dict[TileXY, np.ndarray]
TileOffset = dict[TileXY, Vector]
TileFlowData = tuple[np.ndarray, TileFlow, TileOffset]

def _relative_intersection(
    box1: bounding_box.BoundingBox, box2: bounding_box.BoundingBox
) -> tuple[bounding_box.BoundingBox, bounding_box.BoundingBox]:
  ibox = box1.intersection(box2)
  return (
      bounding_box.BoundingBox(start=ibox.start - box1.start, size=ibox.size),
      bounding_box.BoundingBox(start=ibox.start - box2.start, size=ibox.size),
  )

def compute_flow_map3d(
    tile_map: Mapping[TileXY, Any],
    tile_shape: ShapeXYZ,
    offset_map: np.ndarray,
    axis: int,
    patch_size: Vector = (120, 120, 120),
    stride: Vector = (40, 40, 40),
    batch_size: int = 16,
) -> tuple[TileFlow, TileOffset]:
  """Computes fine flow between two horizontally or vertically adjacent 3d tiles.

  Args:
    tile_map: maps (x, y) tile coordinates to ndarray-like objects storing
      individual tile data; even object should have shape [1, z, y, x] and
      allow standard indexing
    tile_shape: XYZ shape of an individual 3d tile
    offset_map: [3, 1, y, x]-shaped array where the vector spanning the first
      dimension is a coarse XYZ offset between the tiles (x,y) and (x+1,y) or
      (x,y+1)
    axis: axis along which to look for the neighboring tile (0:x, 1:y)
    patch_size: ZYX patch size in pixels
    stride: ZYX stride for the flow map in pixels
    batch_size: number of flow vectors to estimate simultaneously

  Returns:
    tuple of dictionaries:
      (x, y) -> flow array
      (x, y) -> xyz offset at which the following tile was positioned (relative
        to its native position on the grid) before the flow was computed
  """
    
  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  ret, offsets = {}, {}
  grid_yx_shape = offset_map.shape[-2:]
  pad_zyx = np.array(patch_size) // 2 // stride

  for y in range(0, grid_yx_shape[0] - axis):
    for x in range(0, grid_yx_shape[1] - (1 - axis)):
      # Neighbor tile coordinates.
      ny = y + axis
      nx = x + (1 - axis)

      tile_pre = tile_map[(x, y)]
      tile_post = tile_map[(nx, ny)]

      # Offset here comes from a prior process that established the actual
      # overlap between nearby tiles. These offsets are relative to the default
      # grid layout of the tiles with (dx, dy)-stepping.
      offset = offset_map[:, 0, y, x]  # off_[xyz]

      curr_box = bounding_box.BoundingBox(start=(0, 0, 0), size=tile_shape)
      nbor_box = bounding_box.BoundingBox(
          start=(
              offset[0],
              offset[1],
              offset[2],
          ),
          size=tile_shape,
      )
      isec_curr, isec_nbor = _relative_intersection(curr_box, nbor_box)

      # The start coordinate within the preceding tile, in the direction of the
      # tile-tile connection, be aligned to a multiple of stride size.
      overlap = isec_curr.size[axis]  # xyz, scalar

      offset_within_tile = tile_shape[axis] - overlap
      rounded_offset = offset_within_tile // stride[2 - axis] * stride[2 - axis]
      new_overlap = tile_shape[axis] - rounded_offset  # xyz, scalar
      diff = new_overlap - overlap  # xyz, scalar

      off = np.zeros([3])
      off[axis] = -diff

      # The starting coordinates in the orthogonal directions should also be
      # at a multiple of stride size.
      for ax in 0, 1, 2:
        if ax == axis:
          continue

        s = stride[2 - axis]

        if isec_curr.start[ax] > 0:
          diff = s * np.round(isec_curr.start[ax] / s) - isec_curr.start[ax]
          off[ax] = diff
        elif isec_nbor.start[ax] > 0:
          diff = s * np.round(isec_nbor.start[ax] / s) - isec_nbor.start[ax]
          off[ax] = -diff

      nbor_box = nbor_box.translate(off)
      isec_curr, isec_nbor = _relative_intersection(curr_box, nbor_box)

      assert np.all(isec_curr.start % s == 0)
      assert np.all(isec_nbor.start % s == 0)

      offset = np.array(nbor_box.start - curr_box.start)
      offset[axis] = -isec_curr.size[axis]
      offsets[(x, y)] = tuple(offset.tolist())

      pre = tile_pre[isec_curr.to_slice4d()].squeeze(axis=0)
      post = tile_post[isec_nbor.to_slice4d()].squeeze(axis=0)

      assert pre.shape == post.shape

      f = mfc.flow_field(
          pre, post, patch_size=patch_size, step=stride, batch_size=batch_size
      )
      ret[(x, y)] = np.pad(
          f, [[0, 0]] + [[p, p - 1] for p in pad_zyx], constant_values=np.nan
      )

  return ret, offsets



def compute_flow_map3d(
    tile_layout, 
    tile_volumes, 
    tile_shape: ShapeXYZ,
    offset_map: np.ndarray,
    axis: int,
    patch_size: Vector = (120, 120, 120),
    stride: Vector = (40, 40, 40),
    batch_size: int = 16,
) -> tuple[TileFlow, TileOffset]:
  """Computes fine flow between two horizontally or vertically adjacent 3d tiles.

  Args:
    tile_map: maps (x, y) tile coordinates to ndarray-like objects storing
      individual tile data; even object should have shape [1, z, y, x] and
      allow standard indexing
    tile_shape: XYZ shape of an individual 3d tile
    offset_map: [3, 1, y, x]-shaped array where the vector spanning the first
      dimension is a coarse XYZ offset between the tiles (x,y) and (x+1,y) or
      (x,y+1)
    axis: axis along which to look for the neighboring tile (0:x, 1:y)
    patch_size: ZYX patch size in pixels
    stride: ZYX stride for the flow map in pixels
    batch_size: number of flow vectors to estimate simultaneously

  Returns:
    tuple of dictionaries:
      (x, y) -> flow array
      (x, y) -> xyz offset at which the following tile was positioned (relative
        to its native position on the grid) before the flow was computed
  """
  
  layout_x, layout_y = tile_layout.shape
  tile_map = {}
  for x in range(layout_x):
      for y in range(layout_y):
          tile_id = tile_layout[(x, y)]
          tile_map[(x, y)] = SyncAdapter(tile_volumes[tile_id], tile_id)

  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  ret, offsets = {}, {}
  grid_yx_shape = offset_map.shape[-2:]
  pad_zyx = np.array(patch_size) // 2 // stride

  for y in range(0, grid_yx_shape[0] - axis):
    for x in range(0, grid_yx_shape[1] - (1 - axis)):
      # Neighbor tile coordinates.
      ny = y + axis
      nx = x + (1 - axis)

      tile_pre = tile_map[(x, y)]
      tile_post = tile_map[(nx, ny)]

      # Offset here comes from a prior process that established the actual
      # overlap between nearby tiles. These offsets are relative to the default
      # grid layout of the tiles with (dx, dy)-stepping.
      offset = offset_map[:, 0, y, x]  # off_[xyz]

      curr_box = bounding_box.BoundingBox(start=(0, 0, 0), size=tile_shape)
      nbor_box = bounding_box.BoundingBox(
          start=(
              offset[0],
              offset[1],
              offset[2],
          ),
          size=tile_shape,
      )
      isec_curr, isec_nbor = _relative_intersection(curr_box, nbor_box)

      # The start coordinate within the preceding tile, in the direction of the
      # tile-tile connection, be aligned to a multiple of stride size.
      overlap = isec_curr.size[axis]  # xyz, scalar

      offset_within_tile = tile_shape[axis] - overlap
      rounded_offset = offset_within_tile // stride[2 - axis] * stride[2 - axis]
      new_overlap = tile_shape[axis] - rounded_offset  # xyz, scalar
      diff = new_overlap - overlap  # xyz, scalar

      off = np.zeros([3])
      off[axis] = -diff

      # The starting coordinates in the orthogonal directions should also be
      # at a multiple of stride size.
      for ax in 0, 1, 2:
        if ax == axis:
          continue

        s = stride[2 - axis]

        if isec_curr.start[ax] > 0:
          diff = s * np.round(isec_curr.start[ax] / s) - isec_curr.start[ax]
          off[ax] = diff
        elif isec_nbor.start[ax] > 0:
          diff = s * np.round(isec_nbor.start[ax] / s) - isec_nbor.start[ax]
          off[ax] = -diff

      nbor_box = nbor_box.translate(off)
      isec_curr, isec_nbor = _relative_intersection(curr_box, nbor_box)

      assert np.all(isec_curr.start % s == 0)
      assert np.all(isec_nbor.start % s == 0)

      offset = np.array(nbor_box.start - curr_box.start)
      offset[axis] = -isec_curr.size[axis]
      offsets[(x, y)] = tuple(offset.tolist())

      pre = tile_pre[isec_curr.to_slice4d()].squeeze(axis=0)
      post = tile_post[isec_nbor.to_slice4d()].squeeze(axis=0)

      assert pre.shape == post.shape

      f = mfc.flow_field(
          pre, post, patch_size=patch_size, step=stride, batch_size=batch_size
      )
      ret[(x, y)] = np.pad(
          f, [[0, 0]] + [[p, p - 1] for p in pad_zyx], constant_values=np.nan
      )

  return ret, offsets