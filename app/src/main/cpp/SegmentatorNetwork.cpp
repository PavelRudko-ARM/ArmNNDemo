#include "SegmentatorNetwork.h"

#include <string>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , "armnn", __VA_ARGS__)

IRuntimePtr createRuntime()
{
    IRuntime::CreationOptions options;
    return IRuntime::Create(options);
}

SegmentatorNetwork::SegmentatorNetwork(const std::vector<uint8_t> &modelData, bool reduceFp32ToFp16) :
        runtime(createRuntime())
{
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    auto network = parser->CreateNetworkFromBinary(modelData);

    auto inputNames = parser->GetSubgraphInputTensorNames(0);
    inputBinding = parser->GetNetworkInputBindingInfo(0, inputNames[0]).first;

    auto outputNames = parser->GetSubgraphOutputTensorNames(0);
    outputBinding = parser->GetNetworkOutputBindingInfo(0, outputNames[0]).first;

    LOGD("Input: %s, Output: %s", inputNames[0].c_str(), outputNames[0].c_str());

    OptimizerOptions optimizerOptions(reduceFp32ToFp16, false);

    IOptimizedNetworkPtr optNet = Optimize(*network, {Compute::GpuAcc}, runtime->GetDeviceSpec(), optimizerOptions);
    runtime->LoadNetwork(networkId, std::move(optNet));
}

SegmentatorNetwork::~SegmentatorNetwork()
{
}

void SegmentatorNetwork::run(float *imageData, float *maskData)
{
    armnn::InputTensors inputTensors{{inputBinding, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId, inputBinding), imageData)}};
    armnn::OutputTensors outputTensors{{outputBinding, armnn::Tensor(runtime->GetOutputTensorInfo(networkId, outputBinding), maskData)}};

    runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);
}
