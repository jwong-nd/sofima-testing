from google.auth import compute_engine
from google.cloud import storage

def write_to_bucket_gcs(bucket_name: str, 
                        blob_name: str,
                        local_path: str,
                        project: str = 'neural-dynamics-dev'):
    """
    Uploads disk data -> cloud.

    bucket_name: public GCS bucket
    blob_name: cloud path
    local_path: data to upload
    """
    credentials = compute_engine.Credentials()
    client = storage.Client(credentials=credentials, project=project)

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    return blob.public_url

def read_from_bucket_gcs(bucket_name: str,
                         blob_name: str,
                         local_path: str,
                         project: str = 'neural-dynamics-dev'):
    """
    Downloads cloud data -> disk.

    bucket_name: public GCS bucket
    blob_name: cloud path to download from
    local_path: disk path
    """

    credentials = compute_engine.Credentials()
    client = storage.Client(credentials=credentials, project=project)

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


# TODO:
# Implement this when it is needed
def write_to_bucket_s3():
    pass 

def read_to_bucket_s3():
    pass