# ViT (Vision Transformer)
Vision transformer implementation with python, pure C++, and CUDA

This repo trains/infers on CIFAR10 dataset, which can be downloaded through torchvision.
The `python` module contains the vanilla vision transformer architecture and its training/inference logic using PyTorch.
The same logic is implemented using pure C++ functions with standard C/C++ libraries and OpenMP for parallelization directives.
Also, in `cuda` module, the same logic is implemented using CUDA for GPU acceleration.
