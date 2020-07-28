#ifndef ARMNN_DEMO_SEGMENTATORNETWORK_H
#define ARMNN_DEMO_SEGMENTATORNETWORK_H

#include <android/log.h>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>

using namespace armnn;

class SegmentatorNetwork
{
public:
    SegmentatorNetwork(const std::vector<uint8_t>& modelData, bool reduceFp32ToFp16);

    SegmentatorNetwork(SegmentatorNetwork&&) = delete;
    SegmentatorNetwork(const SegmentatorNetwork&) = delete;

    ~SegmentatorNetwork();

    void run(float *imageData, float *maskData);

private:
    NetworkId networkId;
    LayerBindingId outputBinding;
    LayerBindingId inputBinding;
    IRuntimePtr runtime;
};

#endif