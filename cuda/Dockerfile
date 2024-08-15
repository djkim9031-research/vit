FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    git \
    g++-9 \
    gcc-9 \
    wget \
    curl \
    unzip \
    libopencv-dev \
    libgtest-dev \
    python3-dev \
    python3-pip &&   \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install cuDNN
RUN mkdir -p /var/cudnn-local-repo && \
    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/12.0/cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb -P /var/cudnn-local-repo && \
    apt install /var/cudnn-local-repo/cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb

# Install Python packages
RUN pip3 install numpy torch==1.13.1 torchvision==0.14.1 matplotlib==3.7.1

# Install CMake 3.29.6
RUN cd /opt && wget https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6.tar.gz && \
    tar -zxvf cmake-3.29.6.tar.gz && \
    cd cmake-3.29.6 && \
    ./bootstrap && \
    make && make install && \
    rm ../cmake-3.29.6.tar.gz 

# Install gtest
RUN cd /usr/src/gtest && \
    cmake . && \
    make && \
    cp ./lib/libgtest*.a /usr/lib

# Install cudnn-frontend (cudnn includes/libs are located in python dist-packaged!)
RUN cp -r /usr/local/lib/python3.10/dist-package/nvidia/cudnn /usr/local/cuda

RUN git clone --recursive https://github.com/NVIDIA/cudnn-frontend.git && \
    cd cudnn-frontend && \
    mkdir build && cd build && \
    cmake -DCUDNN_PATH=/usr/local/cuda/cudnn -DCUDAToolkit_ROOT=/usr/local/cuda -DCUDNN_FRONTEND_BUILD_SAMPLES=OFF -DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=OFF .. && \
    cmake --build . -j16 && \
    make install

# Set the working directory in the container
WORKDIR /code

# Copy local files under the current dir into the container
COPY . /code

# Set the default command to bash
CMD ["/bin/bash"]