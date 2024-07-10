FROM ubuntu:20.04

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

# Set the working directory in the container
WORKDIR /code

# Copy local files under the current dir into the container
COPY . /code

# Set the default command to bash
CMD ["/bin/bash"]