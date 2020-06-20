#include <jni.h>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "ClassifierNetwork.h"

ClassifierNetwork* gNetwork = nullptr;

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Classifier_prepareNetwork(JNIEnv *env, jobject thiz, jobject javaAssetManager)
{
    // We need to load the network asset into the memory with the AssetManager provided by Activity
    auto assetManager = AAssetManager_fromJava(env, javaAssetManager);

    AAsset *modelAsset = AAssetManager_open(assetManager, "mobilenet_v2.tflite", AASSET_MODE_UNKNOWN);
    off_t bufferSize = AAsset_getLength(modelAsset);
    std::vector<uint8_t> modelData(bufferSize);
    AAsset_read(modelAsset, modelData.data(), bufferSize);
    AAsset_close(modelAsset);

    gNetwork = new ClassifierNetwork(modelData);
}

extern "C" JNIEXPORT int JNICALL Java_com_arm_armnn_1demo_Classifier_runNetwork(JNIEnv *env, jobject thiz, jfloatArray imageData)
{
    jfloat *elements =  env->GetFloatArrayElements(imageData, 0);
    return gNetwork->run(elements);
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Classifier_cleanupNetwork(JNIEnv *env, jobject thiz)
{
    delete gNetwork;
}