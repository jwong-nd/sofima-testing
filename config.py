import yaml
from pathlib import Path 

class PipelineConfiguration(): 
    """
    ALL SOFIMA INTERMEDIATE/FINAL OUTPUTS
    STORED IN CLOUD FOLDER WITH FOLLOWING NAMING CONVENTION:
    {bucket}/SOFIMA_{dataset}/{PipelineConfiguration.name}
    """

    def __init__(self, config_file: str = '/home/jonathan.wong/sofima-testing/config.yml'): 
        # Read yaml file
        with open(config_file, 'r') as file:
            self.params = yaml.safe_load(file)

        # Define standardized names/paths of intermediate/final outputs
        self.dataset_name = Path(self.params['input_dataset_path']).parent
        PipelineConfiguration.COARSE_MESH_NAME = f'coarse_mesh_{self.dataset_name}.npz'
        PipelineConfiguration.ELASTIC_MESH_NAME = f'fine_mesh_{self.dataset_name}.npz'
        PipelineConfiguration.FULL_RES_NAME = f'full_res_{self.dataset_name}.zarr'
        PipelineConfiguration.FUSION_NAME = f'multiscale_{self.dataset_name}.zarr'