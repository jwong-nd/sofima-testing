import tensorstore as ts 
import zarr
from ome_zarr import writer

def open_zarr(bucket: str, path: str):
    return ts.open({
    'driver': 'zarr',
    'kvstore': {
        'driver': 'gcs',
        'bucket': bucket,
    },
    'path': path,
    }).result()


# Metadata copied from original input zarr except uint16 -> uint8
# def write_zarr(bucket: str, shape: list, path: str = 'processed.zarr'): 
#     return ts.open({
#         'driver': 'zarr', 
#         'dtype': 'uint8',
#         'kvstore' : {
#             'driver': 'gcs', 
#             'bucket': bucket,
#         }, 
#         'create': True,
#         'delete_existing': True, 
#         'path': path, 
#         'metadata': {
#         'chunks': [1, 1, 128, 256, 256],
#         'compressor': {
#           'blocksize': 0,
#           'clevel': 1,
#           'cname': 'zstd',
#           'id': 'blosc',
#           'shuffle': 1,
#         },
#         'dimension_separator': '/',
#         'dtype': '<u2',
#         'fill_value': 0,
#         'filters': None,
#         'order': 'C',
#         'shape': shape,  # Ex: [1, 1, 3551, 576, 576]
#         'zarr_format': 2
#         }
#     }).result()


# Tweaking the json schema: 
def write_zarr(bucket: str, shape: list, path: str = 'processed.zarr'): 
    return ts.open({
        'driver': 'zarr', 
        'dtype': 'uint8',
        'kvstore' : {
            'driver': 'gcs', 
            'bucket': bucket,
        }, 
        'create': True,
        'delete_existing': True, 
        'path': path, 
        'metadata': {
        'chunks': [1, 1, 128, 256, 256],
        'compressor': {
          'blocksize': 0,
          'clevel': 1,
          'cname': 'zstd',
          'id': 'blosc',
          'shuffle': 1,
        },
        'dimension_separator': '/',
        'dtype': "|u1",
        'fill_value': 0,
        'filters': None,
        'order': 'C',
        'shape': shape,  # Ex: [1, 1, 3551, 576, 576]
        'zarr_format': 2
        }
    }).result()