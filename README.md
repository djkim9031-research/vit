# ViT (Vision Transformer)
Vision transformer implementation with python, pure C++, and CUDA

This repo trains/infers on CIFAR10 dataset, which can be downloaded through torchvision.
The `python` module contains the vanilla vision transformer architecture and its training/inference logic using PyTorch.
The same logic is implemented using pure C++ functions with standard C/C++ libraries and OpenMP for parallelization directives.
Also, in `cuda` module, the same logic is implemented using CUDA for GPU acceleration.


## How to build
Ubuntu 22.04 with CUDA 12.0 or above is recommended.


### Without CUDA GPU device
Install dockerfile from the main directory and run the docker volume
```
docker build -t vit:latest .
docker run -v ./:/code -it --name vit_volume vit:latest
```

Make bulid directory and navigate there to compile,
```
mkdir build && cd build
cmake ..
make -j${nproc}
```

### With CUDA GPU device
Replace the main directory's dockerfile with the dockerfile from the `cuda` directory, then build & run the docker volume
```
docker build -t vit_cuda:latest .
docker run -v ./:/code -it --name vit_cuda_volume vit_cuda:latest
```

Make bulid directory and navigate there to compile,
```
mkdir build && cd build
cmake ..
make -j${nproc}
```

## How to use

### Python module
For training,
To change the model config, update the `config` field within `train.py` accordingly.
```
python3 ./python/train.py
```

### C++ module [Development 95% done]

First, update the `vit_config.yaml` accordingly based on available compute resources.
Then, to train the model from scratch, (inside the build folder)
```
./vit "path/to/dataset/parent_dir/" "path/to/save_weights" "" "num_train_data_to_read" "num_test_data_to_read"
```

So for example,
```
./vit "../data/" "../data/model_weights" "" 25600 1280
```
The maximum number of train data for CIFAR10 dataset is 50,000 and 10,000 for testset.
It is highly recommended to train with at least >10,000 train dataset and >1,000 test set for robust training performance.

To resume tranining from checkpoint,
```
./vit "path/to/dataset/parent_dir/" "path/to/save_weights" "last_saved_epoch" "num_train_data_to_read" "num_test_data_to_read"
```
So for example,
```
./vit "../data/" "../data/model_weights" 20 25600 1280
```


### CUDA module [under development]
For training,
```
./cuda/vit_cuda
```




## Disclaimer
Depending on your cuda toolkit version, you may have to change the first line of Dockerfile to your specific cuda version.
