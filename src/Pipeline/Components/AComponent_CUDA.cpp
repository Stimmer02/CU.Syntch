#include "AComponent_CUDA.h"

using namespace pipeline;

AComponent_CUDA::AComponent_CUDA(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames, component_type type): type(type), settings(settingCount, settingNames), audioInfo(audioInfo){
    includedIn = nullptr;
}

void AComponent_CUDA::set(uint index, float value){
    settings.values[index] = value;
    settings.copyToDevice();
}

const componentSettings_CUDA* AComponent_CUDA::getSettings(){
    return &settings;
}
