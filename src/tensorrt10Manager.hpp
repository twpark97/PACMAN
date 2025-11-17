#pragma once
// https://github.com/cyrusbehr/tensorrt-cpp-api/blob/main/src/engine.cpp

#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"
#include <cuda_runtime.h>

#include "common.hpp"

namespace paacman{


inline void checkCudaErrorCode(cudaError_t code);

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override;
};


class Tensorrt10Manager
{
public:
  Tensorrt10Manager(const std::string& m_engineName);
  bool runInference(const cv::Mat& image_cpu,
    std::vector<std::vector<float>>& featureVectors);

private:
  inline void checkCudaErrorCode(cudaError_t code);
  void clearGpuBuffers();

private:
  const int GPU_INDEX = 0; // i didn't test for GPU_INDEX > 0
  std::vector<void *> m_buffers;
  std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

  std::vector<uint32_t> m_outputLengths{};
  std::vector<nvinfer1::Dims4> m_inputDims;
  std::vector<nvinfer1::Dims> m_outputDims;
  std::vector<std::string> m_IOTensorNames;

  Logger m_logger;
  cudaStream_t stream_;
};

} // namespace paacman