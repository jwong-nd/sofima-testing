"""Object Wrapper around SOFIMA on Zarr Datasets."""

import functools as ft
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
import time

import zarr_io

from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.volume import subvolume
from sofima import stitch_rigid, flow_utils, stitch_elastic, mesh

import coarse_registration
import fine_registration
import fusion
# ^All are forks



# NOTE:
# - SOFIMA/ZarrStitcher follows following basis convention:  
# o -- x
# |
# y
# Any reference to 'x' or 'y' adopt this basis. 

# - All displacements are defined in pixel space established 
# by the downsample_exp/resolution of the input images. 


class SyncAdapter:
  """Makes it possible to use a TensorStore objects as a numpy array.
  As opposed to .result() the entire tensorstore, 
  this splices the tensorstore, then .result() under numpy syntax.
  """
  
  def __init__(self, tstore):
    self.tstore = tstore

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


class ZarrFusion(fusion.StitchAndRender3dTiles):
    cache = {}
    
    def __init__(self, 
                 tile_volumes: list[ts.Tensorstore],
                 tile_layout: np.ndarray,
                 tile_mesh: np.ndarray,
                 key_to_mesh_index: dict[], # TODO
                 stride_zyx: tuple[int, int, int],
                 offset_xyz: tuple[float, float, float],
                 parallelism=16) -> None:
        super.__init__(self, 
                       tile_layout,
                       tile_mesh,
                       key_to_mesh_index, 
                       stride_zyx,
                       offset_xyz,
                       parallelism)
        self.tile_volumes = tile_volumes
     
    def _open_tile_volume(self, tile_id: int):
        """
        Custom data loading for fusion.
        """
        if tile_id in self.cache:
            return self.cache[tile_id]

        tile = self.tile_volumes[tile_id]
        self.cache[tile_id] = SyncAdapter(tile[0,0,:,:,:])
        return self.cache[tile_id]


class ZarrStitcher: 
    def __init__(self,
                 cloud_storage: str,
                 bucket: str, 
                 dataset_path: str, 
                 tile_names: list[str],
                 downsample_exp: int,
                 tile_layout: np.ndarray) -> None:
        """
        cloud_storage: 'gcs' or 's3'
        bucket: name of bucket
        dataset_path: path from bucket to dataset location
        tile_names: list of tile names inside dataset. 
            Tile index is defines a tile id expected in tile_layout. 
        downsample_exp: Downsample scale to read and operate on. 
        tile_layout: 2D array of tile ids defining relative tile placement. 
        """
        
        self.cloud_storage = cloud_storage
        self.read_bucket = bucket
        self.tile_names = tile_names
        self.downsample_exp = downsample_exp

        # Main Data Structures
        self.tile_volumes: list[ts.Tensorstore] = []  # 5D tczyx homogenous shape
        self.tile_layout = tile_layout
        self.tile_map: dict[tuple[int, int], ts.Tensorstore] = {}

        # Init tile_volumes, init tile size  
        def load_zarr(bucket, tile_location) -> ts.Tensorstore:
            if cloud_storage == 's3':
                return zarr_io.open_zarr_s3(bucket, tile_location)
            else:  # cloud == 'gcs'
                return zarr_io.open_zarr_gcs(bucket, tile_location)
        
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        for t_name in tile_names:
            tile_location = f"{dataset_path}/{t_name}/{downsample_exp}"
            tile = load_zarr(bucket, tile_location)
            self.tile_volumes.append(tile)
            
            _, _, tz, ty, tx = tile.shape
            min_x, min_y, min_z = np.min(min_x, tx), np.min(min_y, ty), np.min(min_z, tz)
        self.tile_size_xyz = min_x, min_y, min_z

        # Standardize size of tile volumes
        for i, tile_vol in enumerate(self.tile_volumes):
            self.tile_volumes[i] = tile_vol[:, :, :min_z, :min_y, :min_x]
        
        # Init tile_map
        for y, row in enumerate(tile_layout):
            for x, tile_id in enumerate(row):
                self.tile_map[(x, y)] = self.tile_volumes[tile_id]


    def run_coarse_registration(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs coarse registration. 
        Returns: 
        cx: tile_layout shape
            Each entry represents displacement of current tile towards right neighbor.
        cy: tile_layout shape
            Each entry represents displacement of current tile towards bottom neighbor. 
        coarse_mesh: (3, 1, tile_layout) shape
            Each entry net displacement of current tile. 
        """

        # Custom data loading for coarse registration
        tile_volumes: list[ts.Tensorstore] = []
        for vol in self.tile_volumes:
            tile_volumes.append(vol.T[:,:,:,0,0])

        cx, cy = coarse_registration.compute_coarse_offsets(self.tile_layout, 
                                                            tile_volumes, 
                                                            False)  # TODO
        coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, 
                                                        cy, 
                                                        mesh_fn=stitch_rigid.elastic_tile_mesh_3d)
        return cx, cy, coarse_mesh

    # Full sofima notebook
    def run_fine_registration(self, 
                              cx: np.ndarray, 
                              cy: np.ndarray, 
                              coarse_mesh: np.ndarray, 
                              stride: tuple[int, int, int]): 
        """
        TODO
        """
        
        # Custom data loading for fine registration
        tile_map = {}
        for key, tstore in self.tile_map.items(): 
            tile_map[key] = SyncAdapter(tstore[0,:,:,:,:])

        # Compute flow map
        flow_x, offsets_x = fine_registration.compute_flow_map3d(tile_map,
                                                                self.tile_size_xyz, 
                                                                cx, axis=0,
                                                                stride=stride,
                                                                patch_size=(80, 80, 80))

        flow_y, offsets_y = fine_registration.compute_flow_map3d(tile_map,
                                                                self.tile_size_xyz, 
                                                                cy, axis=1,
                                                                stride=stride,
                                                                patch_size=(80, 80, 80))

        # Filter patch flows
        kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, "max_deviation": 5, "max_magnitude": 0, "dim": 3}
        fine_x = {k: flow_utils.clean_flow(v, **kwargs) for k, v in flow_x.items()}
        fine_y = {k: flow_utils.clean_flow(v, **kwargs) for k, v in flow_y.items()}

        kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
        fine_x = {k: flow_utils.reconcile_flows([v], **kwargs) for k, v in fine_x.items()}
        fine_y = {k: flow_utils.reconcile_flows([v], **kwargs) for k, v in fine_y.items()}

        # Update mesh (convert coarse tile mesh into fine patch mesh)
        data_x = (cx[:, 0, ...], fine_x, offsets_x)
        data_y = (cy[:, 0, ...], fine_y, offsets_y)
        fx, fy, fine_mesh, nbors, fine_mesh_xy_to_id = fine_registration.aggregate_arrays(
            data_x, data_y, list(tile_map.keys()),
            coarse_mesh[:, 0, ...], stride=stride, tile_shape=self.tile_size_xyz[::-1])

        @jax.jit
        def prev_fn(x):
            target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx, fy=fy, stride=stride)
            x = jax.vmap(target_fn)(nbors)
            return jnp.transpose(x, [1, 0, 2, 3, 4])

        config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                                        num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                        dt_max=100, prefer_orig_order=False,
                                        start_cap=0.1, final_cap=10., remove_drift=True)

        solved_fine_mesh, ekin, t = mesh.relax_mesh(fine_mesh, None, config, prev_fn=prev_fn, mesh_force=mesh.elastic_mesh_3d)

        return solved_fine_mesh, fine_mesh_xy_to_id
    

# (refactor stitch elastic here), Refactor notebook
    def _create_fine_mesh(self, 
                         coarse_mesh: np.ndarray, 
                         stride_zyx: tuple[int, int, int]):
        """
        TODO
        """
        # Return fusion mesh and key to idx

        pass

    def _scale_fine_mesh(self, 
                        fine_mesh: np.ndarray, 
                        fine_mesh_stride_zyx: tuple[int, int, int],
                        scale_factor: int):
        """
        TODO
        """

        # Need to multiply by 4 in all dimensions except the 'linear index'. 
        
        return scaled_fine_mesh, scaled_stride

    def _run_fusion(self, 
                   cloud_storage: str,
                   output_bucket: str, 
                   output_path: str,                    
                   downsample_exp: int,
                   cx: np.ndarray,  # TODO: Subject to change
                   cy: np.ndarray, 
                   fine_mesh: np.ndarray, 
                   fine_mesh_key_to_index: dict[], # TODO
                   stride_zyx: tuple[int, int, int],
                   parallelism: int = 16
                   ): 
        """
        TODO
        """
        if cloud_storage == 's3':
            raise NotImplementedError(
                'Tensorstore does not support s3 writes.'
            )

        self.cloud_storage = cloud_storage
        self.read_bucket = bucket
        self.tile_names = tile_names

        # Depending on downsample exp
        self._scale_fine_mesh()
        # TODO, pass downsample exp to the fusion dataloader. 

        # Approximate fused shape
        cx[np.isnan(cx)] = 0
        cy[np.isnan(cy)] = 0
        x_overlap = cx[2,0,0,0] / self.tile_size_xyz[1]
        y_overlap = cy[1,0,0,0] / self.tile_size_xyz[0]
        y_shape, x_shape = cx.shape[2], cx.shape[3]

        fused_x = self.tile_size_xyz[0] * (1 + ((x_shape - 1) * (1 - x_overlap)))
        fused_y = self.tile_size_xyz[1] * (1 + ((y_shape - 1) * (1 - y_overlap)))
        fused_z = self.tile_size_xyz[2]
        fused_shape = [1, 1, int(fused_z), int(fused_y), int(fused_x)]

        # Calculate crop offset
        # (Need that dense linear index)

        # Perform fusion
        ds_out = zarr_io.write_zarr(bucket, fused_shape, output_path)
        renderer = ZarrFusion(self.tile_volumes, 
                              tile_layout=self.tile_layout, 
                              tile_mesh=fine_mesh, 
                              key_to_mesh_index=fine_mesh_key_to_index,
                              stride_zyx=stride_zyx,
                              offset=crop_offset, 
                              parallelism=parallelism)

        box = bounding_box.BoundingBox(start=(0,0,0), size=ds_out.shape[4:1:-1])  # Needs xyz 
        gen = box_generator.BoxGenerator(box, (512, 512, 512), (0, 0, 0), True) # These are xyz
        renderer.set_effective_subvol_and_overlap((512, 512, 512), (0, 0, 0))
        for i, sub_box in enumerate(gen.boxes):
            t_start = time.time()

            # Feed in an empty subvol, with dimensions of sub_box. 
            inp_subvol = subvolume.Subvolume(np.zeros(sub_box.size[::-1], dtype=np.uint16)[None, ...], sub_box)
            ret_subvol = renderer.process(inp_subvol)  # czyx

            t_render = time.time()

            # ret_subvol is a 4D CZYX volume
            slice = ret_subvol.bbox.to_slice3d()
            slice = (0, 0, slice[0], slice[1], slice[2])
            ds_out[slice].write(ret_subvol.data[0, ...]).result()
            
            t_write = time.time()
            
            print('box {i}: {t1:0.2f} render  {t2:0.2f} write'.format(i=i, t1=t_render - t_start, t2=t_write - t_render))


    def run_fusion_on_coarse_mesh():

        self._create_fine_mesh()
        self._run_fusion()

        pass

    def run_fusion_on_fine_mesh():

        self.run_fusion()

        pass
