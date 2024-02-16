#include "AComponent.h"

using namespace pipeline;

AComponent::AComponent(const audioFormatInfo* audioInfo, uint settingCount, const std::string* settingNames): settings(settingCount, settingNames), audioInfo(audioInfo){
    includedIn = nullptr;
}

void AComponent::set(uint index, float value){
    settings.values[index] = value;
}

const componentSettings* AComponent::getSettings(){
    return &settings;
}
