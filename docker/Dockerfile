FROM nvcr.io/nvidia/tensorrt:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjsoncpp-dev \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --branch 4.8.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.8.0 https://github.com/opencv/opencv_contrib.git

WORKDIR /tmp/opencv/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN="8.9" \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig

ENV TensorRT_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=${TensorRT_DIR}:${LD_LIBRARY_PATH}

WORKDIR /workspace/paacman
COPY . .

RUN mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DTensorRT_DIR=/usr/lib/x86_64-linux-gnu/ \
      -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 \
    && make -j"$(nproc)"

CMD ["/bin/bash"]

