import numpy as np
from ng_link import NgState, link_utils

from sofima.zarr import zarr_io

def _zyx_vector_to_3x4(zyx_vector: np.ndarray):
    output = np.zeros((3, 4))
    
    # Set identity
    output[0, 0] = 1
    output[1, 1] = 1
    output[2, 2] = 1

    # Set translation vector
    output[0, 3] = zyx_vector[2]
    output[1, 3] = zyx_vector[1]
    output[2, 3] = zyx_vector[0]
    return output


def convert_matrix_3x4_to_5x6(matrix_3x4: np.ndarray) -> np.ndarray:
    # Initalize
    matrix_5x6 = np.zeros((5, 6), np.float16)
    np.fill_diagonal(matrix_5x6, 1)

    # Swap Rows 0 and 2; Swap Colums 0 and 2
    patch = np.copy(matrix_3x4)
    patch[[0, 2], :] = patch[[2, 0], :]
    patch[:, [0, 2]] = patch[:, [2, 0]]

    # Place patch in bottom-right corner
    matrix_5x6[2:6, 2:7] = patch

    return matrix_5x6


def apply_deskewing(matrix_3x4: np.ndarray, theta: float = -45) -> np.ndarray:
    # Deskewing
    # X vector => XZ direction
    deskew_factor = np.tan(np.deg2rad(theta))
    deskew = np.array([[1, 0, 0], [0, 1, 0], [deskew_factor, 0, 1]])
    matrix_3x4 = deskew @ matrix_3x4

    return matrix_3x4

def ng_link_single_channel(zd: zarr_io.ZarrDataset,
                        coarse_mesh_to_scale: np.ndarray,
                        max_dr: int = 200,
                        opacity: float = 1.0, 
                        blend: str = "default",
                        output_json_path: str = ".") -> None:
    ng_link_multi_channel([zd],
                          coarse_mesh_to_scale, 
                          max_dr, 
                          opacity, 
                          blend, 
                          output_json_path)

def ng_link_multi_channel(zd_list: list[zarr_io.ZarrDataset],
                        coarse_mesh_to_scale: np.ndarray, 
                        max_dr: int = 200,
                        opacity: float = 1.0, 
                        blend: str = "default",
                        output_json_path: str = ".") -> None:
    """
    zarr_dataset: Zarr Dataset with tile_layout, tile_size, etc. info inside
    coarse_mesh: Output of SOFIMA coarse registration, tile-level flow field
        represented in unified coordinate system. 
    """
    
    # Following neuroglancer convention: 
    # o -- x
    # |
    # y

    # Coarse mesh contains the absolute tile positions
    # coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx_to_scale, 
    #                                                 cy_to_scale, 
    #                                                 mesh_fn=stitch_rigid.elastic_tile_mesh_3d)
    coarse_mesh = coarse_mesh_to_scale

    mesh_y, mesh_x = coarse_mesh.shape[2:]
    tile_positions_xyz = np.zeros((3, mesh_y, mesh_x))
    for yi, y in enumerate(range(mesh_y)):
        for xi, x in enumerate(range(mesh_x)):
            tile_positions_xyz[:, y, x] = coarse_mesh[:, 0, y, x]

    # Generate input config
    zd = zd_list[0]
    layers = []  # Nueroglancer Tabs
    input_config = {
        "dimensions": {
            "x": {"voxel_size": zd.vox_size_xyz[0], "unit": "microns"},
            "y": {"voxel_size": zd.vox_size_xyz[1], "unit": "microns"},
            "z": {"voxel_size": zd.vox_size_xyz[2], "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    for dataset in zd_list:
        # Configure Tab Appearance
        hex_val: int = link_utils.wavelength_to_hex(dataset.channel)
        hex_str = f"#{str(hex(hex_val))[2:]}"

        sources = []  # Tiles within tabs
        layers.append(
            {
                "type": "image",  # Optional
                "source": sources,
                "channel": 0,  # Optional
                "shaderControls": {
                    "normalized": {"range": [0, max_dr]}
                },  # Optional  # Exaspim has low HDR
                "shader": {
                    "color": hex_str,
                    "emitter": "RGB",
                    "vec": "vec3",
                },
                "visible": True,  # Optional
                "opacity": opacity,
                "name": f"CH_{dataset.channel}",
                "blend": blend,
            }
        )

        # Add Tiles to Tab
        for yi in range(zd.tile_layout.shape[0]):
            for xi in range(zd.tile_layout.shape[1]):
                tile_id = dataset.tile_layout[(yi, xi)]
                url = f"s3://{zd.bucket}/{zd.dataset_path}/{dataset.tile_names[tile_id]}"

                tile_id = zd.tile_layout[yi, xi]
                tr_zyx = tile_positions_xyz[:, yi, xi][::-1]
                matrix_3x4 = _zyx_vector_to_3x4(tr_zyx)
                if isinstance(zd, zarr_io.DiSpimDataset): 
                    matrix_3x4 = apply_deskewing(matrix_3x4, zd.theta)
                matrix_5x6 = convert_matrix_3x4_to_5x6(matrix_3x4)
                
                sources.append(
                    {"url": url, "transform_matrix": matrix_5x6.tolist()}
                )
    
    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path="aind-open-data",
        output_json=output_json_path,
    )
    neuroglancer_link.save_state_as_json()
    # print(neuroglancer_link.get_url_link())

    return input_config