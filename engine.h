#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

namespace sample_onnx {

class Engine {
public:
    // Engine(const string &engine_path, bool verbose=false);
    Engine(const char *onnx_model, size_t onnx_size, bool verbose, size_t workspace_size);

    ~Engine();

    void save(const string &path);
    void infer(vector<void *> &buffers, int batch);

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    void _load(const string &path);
    void _prepare();

};

}