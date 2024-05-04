#ifndef STRINGTOENUM_H
#define STRINGTOENUM_H

#include "Pipeline/ComponentManager.h"
#include "Pipeline/Components/AAdvancedComponent.h"
#include "Pipeline/IDManager.h"
#include "Synthesizer_CUDA.h"

#include <stdexcept>
#include <cstring>
#include <map>

namespace pipeline{
    ID_type stringToIDType(const char*& IDTypeString);
    std::string IDTypeToString(ID_type IDType);

    component_type stringToComponentType(const char*& componentTypeString);
    std::string componentTypeToString(component_type compType);

    advanced_component_type stringToAdvComponentType(const char*& componentTypeString);
    std::string advComponentTypeToString(advanced_component_type compType);
}
namespace synthesizer{
    generator_type stringToSynthType(const char*& synthTypeString);
    settings_name stringToSettingName(const char*& settingNameString);
}

#endif
