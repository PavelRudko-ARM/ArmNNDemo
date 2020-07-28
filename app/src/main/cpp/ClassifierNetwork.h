/*#ifndef ARMNN_DEMO_CLASSIFIERNETWORK_H
#define ARMNN_DEMO_CLASSIFIERNETWORK_H

#include <android/log.h>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>

using namespace armnn;

class ClassifierNetwork
{
    public:
        ClassifierNetwork(const std::vector<uint8_t>& modelData);

        ClassifierNetwork(ClassifierNetwork&&) = delete;
        ClassifierNetwork(const ClassifierNetwork&) = delete;

        ~ClassifierNetwork();

        int run(float *imageData);

    private:
        NetworkId networkId;
        LayerBindingId outputBinding;
        IRuntimePtr runtime;
};

#endif*/
