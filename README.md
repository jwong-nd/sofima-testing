# sofima-testing

Sofima testing code. 

## Environment Setup:
0) Create a VM with preinstalled CUDA 11.3 drivers. This will save you a lot of work.

1) Run sofima_env.sh:
```
chmod u+x sofima_env.sh
./sofima_env.sh
```

2) Upload compatible cuDNN package, run sofima_cudnn.sh:
Comaptible cuDNN package: cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
```
chmod u+x sofima_cudnn.sh
./sofima_cudnn.sh
```

Environment setup is a bash script because exporting a env.yml does not work. 
(breaks on environment creation due to pip dependency conflicts)