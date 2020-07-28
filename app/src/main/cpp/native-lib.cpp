#include <jni.h>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "ClassifierNetwork.h"
#include "SegmentatorNetwork.h"

//ClassifierNetwork* gClassifierNetwork = nullptr;
SegmentatorNetwork* gSegmentatorNetwork = nullptr;

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Classifier_prepareNetwork(JNIEnv *env, jobject thiz, jobject javaAssetManager)
{
    /*auto assetManager = AAssetManager_fromJava(env, javaAssetManager);

    AAsset *modelAsset = AAssetManager_open(assetManager, "mobilenet_v2.tflite", AASSET_MODE_UNKNOWN);
    off_t bufferSize = AAsset_getLength(modelAsset);
    std::vector<uint8_t> modelData(bufferSize);
    AAsset_read(modelAsset, modelData.data(), bufferSize);
    AAsset_close(modelAsset);

    gClassifierNetwork = new ClassifierNetwork(modelData);*/
}

extern "C" JNIEXPORT int JNICALL Java_com_arm_armnn_1demo_Classifier_runNetwork(JNIEnv *env, jobject thiz, jfloatArray imageData)
{
    jfloat *elements =  env->GetFloatArrayElements(imageData, 0);
    return 5;
    //return gClassifierNetwork->run(elements);
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Classifier_cleanupNetwork(JNIEnv *env, jobject thiz)
{
    //delete gClassifierNetwork;
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Segmentator_prepareNetwork(JNIEnv *env, jobject thiz, jobject javaAssetManager, jboolean reduceFp32ToFp16)
{
    auto assetManager = AAssetManager_fromJava(env, javaAssetManager);

    AAsset *modelAsset = AAssetManager_open(assetManager, "deconv_fin_munet.tflite", AASSET_MODE_UNKNOWN);
    off_t bufferSize = AAsset_getLength(modelAsset);
    std::vector<uint8_t> modelData(bufferSize);
    AAsset_read(modelAsset, modelData.data(), bufferSize);
    AAsset_close(modelAsset);

    gSegmentatorNetwork = new SegmentatorNetwork(modelData, reduceFp32ToFp16);
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Segmentator_runNetwork(JNIEnv *env, jobject thiz, jfloatArray imageData, jfloatArray maskData)
{
    jfloat *imageDataElements =  env->GetFloatArrayElements(imageData, 0);
    jfloat *maskDataElements =  env->GetFloatArrayElements(maskData, 0);
    gSegmentatorNetwork->run(imageDataElements, maskDataElements);
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_armnn_1demo_Segmentator_cleanupNetwork(JNIEnv *env, jobject thiz)
{
    delete gSegmentatorNetwork;
}