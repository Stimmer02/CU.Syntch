#ifndef STRINGTOENUM_H
#define STRINGTOENUM_H

#include "Pipeline/ComponentManager.h"
#include "Pipeline/IDManager.h"
#include "Synthesizer.h"

#include <stdexcept>
#include <cstring>
#include <map>

namespace pipeline{
    ID_type stringToIDType(const char*& IDTypeString);
    std::string IDTypeToString(ID_type IDType);

    component_type stringTocomponentType(const char*& componentTypeString);
}
namespace synthesizer{
    generator_type stringToSynthType(const char*& synthTypeString);
    settings_name stringToSettingName(const char*& settingNameString);
}

#endif
