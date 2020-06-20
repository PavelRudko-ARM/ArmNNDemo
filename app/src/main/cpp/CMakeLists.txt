cmake_minimum_required(VERSION 3.4.1)

#Set up ArmNN
set(ARMNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../armnn)
include_directories(${ARMNN_DIR}/include)
add_library(armnn SHARED IMPORTED)
add_library(armnnTfLiteParser SHARED IMPORTED)
set_target_properties(armnn PROPERTIES IMPORTED_LOCATION ${ARMNN_DIR}/bin/${ANDROID_ABI}/libarmnn.so)
set_target_properties(armnnTfLiteParser PROPERTIES IMPORTED_LOCATION ${ARMNN_DIR}/bin/${ANDROID_ABI}/libarmnnTfLiteParser.so)

add_library(native-lib SHARED native-lib.cpp ClassifierNetwork.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(native-lib armnn armnnTfLiteParser ${log-lib} ${android-lib})