/*#include "ClassifierNetwork.h"

#include <string>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , "armnn", __VA_ARGS__)

// Just a helper to create this unique_ptr in the member initializer list
IRuntimePtr createRuntime()
{
    IRuntime::CreationOptions options;
    return IRuntime::Create(options);
}

ClassifierNetwork::ClassifierNetwork(const std::vector<uint8_t> &modelData) :
    runtime(createRuntime())
{
    // Parsing the model from memory
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    auto network = parser->CreateNetworkFromBinary(modelData);

    // We need to know the output binding id, it can be found by name
    auto outputNames = parser->GetSubgraphOutputTensorNames(0);
    outputBinding = parser->GetNetworkOutputBindingInfo(0, outputNames[0]).first;

    // Optimizing the network for a particular backend and assigning it to the runtime
    IOptimizedNetworkPtr optNet = Optimize(*network, {Compute::GpuAcc}, runtime->GetDeviceSpec());
    runtime->LoadNetwork(networkId, std::move(optNet));
}

ClassifierNetwork::~ClassifierNetwork()
{
}

int ClassifierNetwork::run(float *imageData)
{
    std::vector<float> outputData(runtime->GetOutputTensorInfo(networkId, outputBinding).GetNumElements());

    // Here we must specify input and output tensors, providing their ids as well
    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId, 0), imageData)}};
    armnn::OutputTensors outputTensors{{outputBinding, armnn::Tensor(runtime->GetOutputTensorInfo(networkId, outputBinding), outputData.data())}};

    runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

    // Choosing the class index with the highest probability
    return std::distance(outputData.begin(), std::max_element(outputData.begin(), outputData.end()));
}
*/