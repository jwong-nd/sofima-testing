# sofima-testing

Sofima testing code. 


## Development Environment Setup Instructions from environment.yml
0) Create a VM with preinstalled CUDA 11.3 drivers. This will save you a lot of work.

1) Setup conda environment: 
    ```
    conda env create -f environment.yml
    ```

2) Run the following commands to install the compatible cuDNN library to this JAX version specified in the environment.yml file:
    ```
    $ tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
    $ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    $ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```

## Development Environment Setup Instructions for Scratch
0) Create a VM with preinstalled CUDA 11.3 drivers. This will save you a lot of work.
1) Create Python 3.11 conda environment
2) Pip install standard packages. Exceptions: 
    - SOFIMA is in active development, pip install the most recent commit: 
    ```
    pip install git+https://github.com/google-research/sofima
    ```
    - Install jaxlib/CUDA: 
    ```
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

3) Last JAX dependency, cuDNN library version, is pulled from the NVIDIA website (you need to first sign up for an NVIDIA account). Downloaded cuDNN library must match the major version and be greater than the minor version of the cuDNN compiled into JAX. (Essentially, just download the newest version.)

Downloaded File looks something like this: ```cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz```
Simply copy the zip file onto the VM, unzip, and copy into base folders:
    ```
    $ tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
    $ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    $ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```

(Step (2) tehcnically downloads cuDNN, but not the same version in which the jax wheel is compiled on. This step installs the same cuDNN verion.)