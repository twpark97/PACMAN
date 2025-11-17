cmake_minimum_required(VERSION 3.10)
project(paacman)

add_compile_options(-std=c++17 -Wall -Wextra -pedantic -O3)

set(TensorRT_DIR /usr/lib/x86_64-linux-gnu/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(TensorRT REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")

include_directories(
  ${jsoncpp_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
)

add_executable(main
  src/main.cpp
  src/paacman.cpp
  src/parameters.cpp
  src/common.cpp
  src/debuggingInfo.cpp
  src/detectedMarker.cpp
  src/tensorrt10Manager.cpp
)

target_link_libraries(main
  jsoncpp
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
)
