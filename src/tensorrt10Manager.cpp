#include <filesystem>
#include <algorithm>

#include "tensorrt10Manager.hpp"

using std::cout;
using std::endl;
using std::string;



namespace paacman{

void Tensorrt10Manager::checkCudaErrorCode(cudaError_t code) {
  if (code != cudaSuccess) {
      std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                            "), with message: " + cudaGetErrorString(code);
      throw std::runtime_error(errMsg);
  }
}

void Tensorrt10Manager::clearGpuBuffers() {
  if (!m_buffers.empty()) {
    // Free GPU memory of outputs
    const auto numInputs = m_inputDims.size();
    for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
      checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
    }
    m_buffers.clear();
  }
}

Tensorrt10Manager::Tensorrt10Manager(const std::string& m_engine_name){
  // Read the serialized model from disk
  if (false == fs::exists(m_engine_name)){
    string msg = "Error, unable to read TensorRT model at path: "  + m_engine_name;
    throw std::runtime_error(msg);
  } 
  else {
    auto msg = "Loading TensorRT engine file at path: " + m_engine_name;
  }

  std::ifstream file(m_engine_name, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    auto msg = "Error, unable to read engine file";
    throw std::runtime_error(msg);
  }

  // Create a runtime to deserialize the engine file.
  m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
  if (!m_runtime) {
    throw std::runtime_error("Unable to create InferRuntime");
  }

  // Set the device index
  auto ret = cudaSetDevice(GPU_INDEX);
  if (ret != 0) {
    checkCudaErrorCode(ret);

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    auto errMsg = "Unable to set GPU device index to: " + std::to_string(GPU_INDEX) + ". Note, your device has " +
                  std::to_string(numGPUs) + " CUDA-capable GPU(s).";
    throw std::runtime_error(errMsg);
  }

  // Create an engine, a representation of the optimized model.
  m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!m_engine) {
    throw std::runtime_error("Unable to create ICudaEngine");
  }

  // The execution context contains all of the state associated with a
  // particular invocation
  m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
  if (!m_context) {
    throw std::runtime_error("Unable to create IExecutionContext");
  }

  // Storage for holding the input and output buffers
  // This will be passed to TensorRT for inference
  m_buffers.resize(m_engine->getNbIOTensors());

  m_outputLengths.clear();
  m_inputDims.clear();
  m_outputDims.clear();
  m_IOTensorNames.clear();

  // Create a cuda stream
  checkCudaErrorCode(cudaStreamCreate(&stream_));

  // Allocate GPU memory for input and output buffers
  m_outputLengths.clear();
  for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
    const auto tensorName = m_engine->getIOTensorName(i);
    m_IOTensorNames.emplace_back(tensorName);
    const auto tensorType = m_engine->getTensorIOMode(tensorName);
    const auto tensorShape = m_engine->getTensorShape(tensorName);
    const auto tensorDataType = m_engine->getTensorDataType(tensorName);

    if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
      if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
          auto msg = "Error, the implementation currently only supports float inputs";
          throw std::runtime_error(msg);
      }

      m_inputDims.emplace_back(
        tensorShape.d[0],
        tensorShape.d[1], 
        tensorShape.d[2], 
        tensorShape.d[3]);
      cout << "input dimension of trt model: (" 
        << m_inputDims[0].d[0] << ' '
        << m_inputDims[0].d[1] << ' '
        << m_inputDims[0].d[2] << ' '
        << m_inputDims[0].d[3] << ")" << endl;

      checkCudaErrorCode(
        cudaMallocAsync(
          &m_buffers[i], 
          tensorShape.d[0] * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float),
          stream_));
    } 
    else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
      // The binding is an output
      uint32_t outputLength = 1;
      m_outputDims.push_back(tensorShape);
      
      for (int j = 0; j < tensorShape.nbDims; ++j) {
        outputLength *= tensorShape.d[j];
      }

      cout << "output dimension of trt model: "
        << i << ' ' 
        << outputLength << " (" 
        << tensorShape.d[0] << ' '
        << tensorShape.d[1] << ' '
        << tensorShape.d[2] << ")" << endl;

      m_outputLengths.push_back(outputLength);
      checkCudaErrorCode(
        cudaMallocAsync(
          &m_buffers[i], 
          outputLength * sizeof(float),
          stream_));
    } 
    else {
      auto msg = "Error, IO Tensor is neither an input or output!";
      throw std::runtime_error(msg);
    }
  }

  // Synchronize and destroy the cuda stream
  checkCudaErrorCode(cudaStreamSynchronize(stream_));
}

void Logger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
      std::cout << msg << std::endl;
  }
}

bool Tensorrt10Manager::runInference(const cv::Mat& image_cpu,
    std::vector<std::vector<float>>& featureVectors)
{
  if (image_cpu.rows == 0 || image_cpu.cols == 0){
    cout << "No inputs provided to runInference!" << endl;
    return false;
  }
  
  cv::Mat image_cpu_float;
  image_cpu.convertTo(image_cpu_float, CV_32FC1, 1.f / 255.f);
  
  if (false == image_cpu_float.isContinuous())
    image_cpu_float = image_cpu_float.clone();
  
  checkCudaErrorCode(
    cudaMemcpyAsync(
      m_buffers[0], 
      image_cpu_float.ptr<float>(), 
      image_cpu_float.rows * image_cpu_float.cols * sizeof(float),
      cudaMemcpyHostToDevice, 
      stream_)
  );

  // Ensure all dynamic bindings have been defined.
  if (!m_context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    throw std::runtime_error(msg);
  }

  // Set the address of the input and output buffers
  for (size_t i = 0; i < m_buffers.size(); ++i) {
    bool status = m_context->setTensorAddress(
      m_IOTensorNames[i].c_str(),
      m_buffers[i]);
    if (!status) {
      auto msg =  "Error, unable to set tensor address!";
      throw std::runtime_error(msg);
    }
  }

  // Run inference.
  bool status = m_context->enqueueV3(stream_);
  if (!status) {
    cout << "inference failed! " << endl;
    return false;
  }

  // Copy the outputs back to CPU
  if (featureVectors.size() != m_outputLengths.size())
    featureVectors.resize(m_outputLengths.size());
  for (size_t i = 0; i < featureVectors.size(); ++i) {
    if (featureVectors[i].size() != m_outputLengths[i]){
      featureVectors[i].resize(m_outputLengths[i]);
    }
  }

  int numInputs = 1;
  for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {

    // Copy the output
    float outputLenFloat = m_outputLengths[outputBinding - numInputs];
    checkCudaErrorCode(
      cudaMemcpyAsync(
        featureVectors[outputBinding - numInputs].data(), 
        static_cast<char*>(m_buffers[outputBinding]), 
        outputLenFloat * sizeof(float),
        cudaMemcpyDeviceToHost, 
        stream_
      )
    );
  }

  

  // Synchronize the cuda stream
  checkCudaErrorCode(cudaStreamSynchronize(stream_));
  return true;
}
}
