"""'Fork' of sofima.processor.warp."""

from concurrent import futures
from typing import Any, Sequence

from absl import logging
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import edt
import numpy as np
from sofima import map_utils
from sofima import warp


ZYX = tuple[int, int, int]
XYZ = tuple[int, int, int]


class StitchAndRender3dTiles(subvolume_processor.SubvolumeProcessor):
  """Renders a volume by stitching 3d tiles placed on a 2d grid."""

  _tile_meshes = None
  _mesh_index_to_xy = {}
  _tile_boxes = {}
  _inverted_meshes = {}

  crop_at_borders = False

  def __init__(
      self,
      tile_layout: Sequence[Sequence[int]],
      tile_mesh: str,
      xy_to_mesh_index: dict[int, tuple],
      stride: ZYX,
      offset: XYZ = (0, 0, 0),
      margin: int = 0,
      work_size: XYZ = (128, 128, 128),
      order: int = 1,
      parallelism: int = 16
  ):
    """Constructor.

    Args:
      tile_map: yx-shaped grid of tile IDs
      tile_idx_to_xy: index 
      stride: ZYX stride of the mesh in pixels
      offset: XYZ global offset to apply to the rendered image
      margin: number of pixels away from the tile boundary to ignore during
        rendering; this can be useful is the tiles are more distorted at their
        boundaries. Do not apply to the outer edges of tiles at the edges of the
        grid.
      work_size: see `warp.ndimage_warp`
      order: see `warp.ndimage_warp`
      parallelism: see `warp.ndimage_warp`
    """
    self._tile_layout = tile_layout
    self._stride = stride
    self._offset = offset
    self._margin = margin
    self._order = order
    self._parallelism = parallelism
    self._work_size = work_size

    StitchAndRender3dTiles._tile_meshes = tile_mesh
    StitchAndRender3dTiles._mesh_index_to_xy = {
      v:k for k, v in xy_to_mesh_index.items()
    }
    assert StitchAndRender3dTiles._tile_meshes.shape[1] == len(
        StitchAndRender3dTiles._mesh_index_to_xy
    )
    
    self._xy_to_tile_id = {}
    for y, row in enumerate(tile_layout):
      for x, tile_id in enumerate(row):
        self._xy_to_tile_id[(x, y)] = tile_id

  def _open_tile_volume(self, tile_id: int) -> Any:
    """Returns a ZYX-shaped ndarray-like object representing the tile data."""
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def context(self):
    return (0, 0, 0), (0, 0, 0)

  def _collect_tile_boxes(self, tile_shape_zyx: ZYX):
    map_box = bounding_box.BoundingBox(
        start=(0, 0, 0),
        size=StitchAndRender3dTiles._tile_meshes.shape[2:][::-1],
    )

    for i in range(StitchAndRender3dTiles._tile_meshes.shape[1]):
      tx, ty = StitchAndRender3dTiles._mesh_index_to_xy[i]

      mesh = StitchAndRender3dTiles._tile_meshes[:, i, ...]
      tg_box = map_utils.outer_box(mesh, map_box, self._stride)

      # Region that can be rendered with the current tile, in global
      # coordinates.
      out_box = bounding_box.BoundingBox(
          start=(
              tg_box.start[0] * self._stride[2]
              + tx * tile_shape_zyx[-1]
              + self._offset[0],
              tg_box.start[1] * self._stride[1]
              + ty * tile_shape_zyx[-2]
              + self._offset[1],
              tg_box.start[2] * self._stride[0] + self._offset[2],
          ),
          size=(
              tg_box.size[0] * self._stride[2],
              tg_box.size[1] * self._stride[1],
              tg_box.size[2] * self._stride[0],
          ),
      )

      StitchAndRender3dTiles._tile_boxes[i] = out_box, tg_box

  def _get_dts(self, shape: ZYX, tx: int, ty: int) -> np.ndarray:
    # Ignore up to _margin pixels on tile edges, with the exception of the
    # tiles at the outer sides of the tile grid.
    mask = np.zeros(shape[1:], dtype=bool)
    if self._margin > 0:
      x0 = self._margin if tx > 0 else 0
      x1 = -self._margin if tx < self._tile_layout.shape[-1] - 1 else -1
      y0 = self._margin if ty > 0 else 0
      y1 = -self._margin if ty < self._tile_layout.shape[-2] - 1 else -1
      mask[y0:y1, x0:x1] = 1
    else:
      mask[...] = 1

    # Compute a (2d) distance transform of the mask, for use in blending.
    return edt.edt(mask, black_border=True, parallel=0)

  def _load_tile_images(
      self,
      box: bounding_box.BoundingBox,
      tile_shape_zyx: ZYX,
      volstores: dict[int, Any],
      tpe: futures.Executor,
  ) -> set[futures.Future[tuple[np.ndarray, Any]]]:
    fs = set([])

    # Bounding boxes for the tile and its mesh in its own coordinate system
    # (with the tile placed at the origin).
    image_box = bounding_box.BoundingBox(
        start=(0, 0, 0), size=tile_shape_zyx[::-1]
    )
    map_box = bounding_box.BoundingBox(
        start=(0, 0, 0),
        size=StitchAndRender3dTiles._tile_meshes.shape[2:][::-1],
    )

    for i, (out_box, tg_box) in StitchAndRender3dTiles._tile_boxes.items():
      sub_box = out_box.intersection(box)
      if sub_box is None:
        continue

      logging.info('Processing source %r (%r)', i, out_box)

      coord_map = StitchAndRender3dTiles._tile_meshes[:, i, ...]
      tx, ty = StitchAndRender3dTiles._mesh_index_to_xy[i]

      if i not in StitchAndRender3dTiles._inverted_meshes:
        # Add context to avoid rounding issues in map inversion.
        tg_box = tg_box.adjusted_by(start=(-1, -1, -1), end=(1, 1, 1))
        inverted_map = map_utils.invert_map(
            coord_map, map_box, tg_box, stride=self._stride
        )
        # Extrapolate only. The inverted map should not have any holes that
        # can be filled through interpolation.
        inverted_map = map_utils.fill_missing(
            inverted_map, extrapolate=True, interpolate_first=False
        )
        StitchAndRender3dTiles._inverted_meshes[i] = tg_box, inverted_map
      else:
        tg_box, inverted_map = StitchAndRender3dTiles._inverted_meshes[i]

      # Box which can be passed to ndimage_warp to render the *whole* tile.
      # This is within a coordinate system where the source tile is
      # placed at (0, 0, 0).
      local_out_box = out_box.translate((
          -tx * tile_shape_zyx[-1] - self._offset[0],
          -ty * tile_shape_zyx[-2] - self._offset[1],
          -self._offset[2],
      ))

      # Part of the region we can render with the current tile that is
      # actually needed for the current output.
      local_rel_box = sub_box.translate(-out_box.start)
      local_warp_box = local_rel_box.translate(local_out_box.start)

      # Part of the inverted mesh that is needed to render the current region of interest.
      s = 1.0 / np.array(self._stride)[::-1]
      local_map_box = local_warp_box.scale(s).adjusted_by(
          start=(-2, -2, -2), end=(2, 2, 2)
      )
      local_map_box = local_map_box.intersection(tg_box)
      if local_map_box is None:
        continue

      map_query_box = local_map_box.translate(-tg_box.start)
      assert np.all(map_query_box.start >= 0)
      sub_map = inverted_map[map_query_box.to_slice4d()]

      # Part of the source image needed to render the current region
      # of interest.
      data_box = map_utils.outer_box(sub_map, local_map_box, self._stride, 1)
      data_box = data_box.intersection(image_box)
      if data_box is None:
        continue

      dts_2d = self._get_dts(tile_shape_zyx, tx, ty)
      sub_dts = dts_2d[data_box.to_slice_tuple(0, 2)][None, ...]
      sub_dts = np.repeat(sub_dts, data_box.size[2], axis=0)

      # Schedule data loading.
      context = inverted_map, tg_box, local_warp_box, sub_box, sub_dts, data_box
      def _load(context=context, i=i):
        data_box = context[-1]
        image = volstores[i][data_box.to_slice3d()]
        return image, context

      fs.add(tpe.submit(_load))

    return fs

  def process(
      self, subvol: subvolume.Subvolume
  ) -> subvolume_processor.SubvolumeOrMany:
    box = subvol.bbox
    logging.info('Processing %r', box)

    volstores = {}
    for i in range(StitchAndRender3dTiles._tile_meshes.shape[1]):
      tile_id = self._xy_to_tile_id[StitchAndRender3dTiles._mesh_index_to_xy[i]]
      volstores[i] = self._open_tile_volume(tile_id)

    # Bounding boxes representing a single tile placed the origin.
    tile_shape_zyx = next(iter(volstores.values())).shape
    self._collect_tile_boxes(tile_shape_zyx)

    # For blending, accumulate (weighted) image data as floats. This will
    # be normalized and cast to the desired output type once the image is
    # rendered.
    img = np.zeros(subvol.data.shape[1:], dtype=np.float32)
    norm = np.zeros(subvol.data.shape[1:], dtype=np.float32)

    with futures.ThreadPoolExecutor(max_workers=2) as tpe:
      fs = self._load_tile_images(box, tile_shape_zyx, volstores, tpe)

      for f in futures.as_completed(fs):
        image, (
            inverted_map,
            tg_box,
            local_warp_box,
            sub_box,
            sub_dts,
            data_box,
        ) = f.result()

        image = warp.ndimage_warp(
            image,
            inverted_map,
            self._stride,
            work_size=self._work_size,
            overlap=(0, 0, 0),
            order=self._order,
            image_box=data_box,
            map_box=tg_box,
            out_box=local_warp_box,
            parallelism=self._parallelism,
        )

        warped_dts = warp.ndimage_warp(
            sub_dts,
            inverted_map,
            self._stride,
            work_size=self._work_size,
            overlap=(0, 0, 0),
            image_box=data_box,
            map_box=tg_box,
            out_box=local_warp_box,
            parallelism=self._parallelism,
        )

        out_rel_box = sub_box.translate(-box.start)

        img[out_rel_box.to_slice3d()] += image * warped_dts
        norm[out_rel_box.to_slice3d()] += warped_dts

    # Compute the (distance-from-tile-center-) weighted average of every
    # voxel. This results in smooth transitions between tiles, even if
    # there are some contrast differences.
    ret = img
    ret[norm > 0] /= norm[norm > 0]
    ret = ret.astype(self.output_type(subvol.data.dtype))

    return self.crop_box_and_data(box, ret[None, ...])
