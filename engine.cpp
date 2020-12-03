#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <string>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include "engine.h"

using namespace nvinfer1;
using namespace nvonnxparser;

namespace sample_onnx {

class Logger : public ILogger {
public:
    Logger(bool verbose)
        : _verbose(verbose) {
    }

    void log(Severity severity, const char *msg) override {
        if (_verbose || (severity != Severity::kINFO) && (severity != Severity::kVERBOSE))
            cout << msg << endl;
    }

private:
   bool _verbose{false};
};

void Engine::_load(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, file.end);
    size_t size = file.tellg();
    file.seekg (0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

void Engine::_prepare() {
    _context = _engine->createExecutionContext();
    _context->setOptimizationProfile(0);
    cudaStreamCreate(&_stream);
}

Engine::Engine(const string &engine_path, bool verbose) {
    Logger logger(verbose);
    _runtime = createInferRuntime(logger);
    _load(engine_path);
    _prepare();

    mInputDims = _engine->getBindingDimensions(0);
    mOutputDims = _engine->getBindingDimensions(1);
}

Engine::~Engine() {
    if (_stream) cudaStreamDestroy(_stream);
    if (_context) _context->destroy();
    if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();
}

Engine::Engine(const char *onnx_model, size_t onnx_size, bool verbose, size_t workspace_size){
    
    Logger logger(verbose);

    // Create builder
    auto builder = createInferBuilder(logger);
    auto builderConfig = builder->createBuilderConfig();
    builderConfig->setFlag(BuilderFlag::kFP16);
    builderConfig->setMaxWorkspaceSize(workspace_size);

    // Parse ONNX
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto parser = createParser(*network, logger);
    parser->parse(onnx_model, onnx_size);

    // Build engine
    cout << "Applying optimizations and building TRT CUDA engine..." << endl;
    _engine = builder->buildEngineWithConfig(*network, *builderConfig);

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    // Housekeeping
    parser->destroy();
    network->destroy();
    builderConfig->destroy();
    builder->destroy();

    _prepare();

}

void Engine::save(const string &path) {
    cout << "Writing to " << path << "..." << endl;
    auto serialized = _engine->serialize();
    ofstream file(path, ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());

    serialized->destroy();
}

void Engine::infer(vector<void *> &buffers, int batch){
    auto dims = _engine->getBindingDimensions(0);
    _context->setBindingDimensions(0, Dims4(batch, dims.d[1], dims.d[2], dims.d[3]));
    _context->enqueueV2(buffers.data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
}

std::vector<float> Engine::processInput(const string &path) {
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<uint8_t> fileData(inputH * inputW);
    vector<float> data (inputH * inputW);
    readPGMFile(path, fileData.data(), inputH, inputW);
    
    for (int i = 0; i < inputH * inputW; i++)
    {
        data[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return data;
}

}