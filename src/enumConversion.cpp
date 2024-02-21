#include "enumConversion.h"
#include "Pipeline/ComponentManager.h"

pipeline::ID_type pipeline::stringToIDType(const char*& IDTypeString){
    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };
    static std::map<const char*, pipeline::ID_type, cmp_str> IDTypeMap{
        {"synth",       pipeline::SYNTH},
        {"SYNTH",       pipeline::SYNTH},
        {"synthesizer", pipeline::SYNTH},
        {"SYNTHESIZER", pipeline::SYNTH},

        {"input",       pipeline::INPUT},
        {"INPUT",       pipeline::INPUT},

        {"comp",        pipeline::COMP},
        {"COMP",        pipeline::COMP},
        {"component",   pipeline::COMP},
        {"COMPONENT",   pipeline::COMP},

        {"invalid",     pipeline::INVALID},
        {"INVALID",     pipeline::INVALID},
    };

    try {
        return IDTypeMap.at(IDTypeString);
    } catch (std::out_of_range const&){
        return pipeline::INVALID;
    }

    return pipeline::INVALID;
}

std::string pipeline::IDTypeToString(ID_type IDType){
    switch (IDType) {

    case INVALID:
        return "INVALID";
    case INPUT:
        return "INPUT";
    case SYNTH:
        return "SYNTH";
    case COMP:
        return "COMP";
    }
    return "INVALID";
}

pipeline::component_type pipeline::stringToComponentType(const char*& componentTypeString){
    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };
    static std::map<const char*, pipeline::component_type, cmp_str> componentTypeMap{
        {"volume",       pipeline::COMP_VOLUME},
        {"vol",          pipeline::COMP_VOLUME},
        {"VOLUME",       pipeline::COMP_VOLUME},
        {"VOL",          pipeline::COMP_VOLUME},
        {"COMP_VOLUME",  pipeline::COMP_VOLUME},

        {"invalid",      pipeline::COMP_INVALID},
        {"INVALID",      pipeline::COMP_INVALID},
        {"COMP_INVALID", pipeline::COMP_INVALID},
    };

    try {
        return componentTypeMap.at(componentTypeString);
    } catch (std::out_of_range const&){
        return pipeline::COMP_INVALID;
    }

    return pipeline::COMP_INVALID;
}

std::string pipeline::componentTypeToString(component_type compType){
    switch (compType){
        case COMP_INVALID:
            return "COMP_INVALID";
        case COMP_VOLUME:
            return "COMP_VOLUME";
    }
    return "INVALID";
}

pipeline::advanced_component_type pipeline::stringToAdvComponentType(const char*& componentTypeString){
    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };
    static std::map<const char*, pipeline::advanced_component_type, cmp_str> advComponentTypeMap{
        {"sum",  pipeline::ACOMP_SUM2},
        {"SUM",  pipeline::ACOMP_SUM2},
        {"sum2", pipeline::ACOMP_SUM2},
        {"SUM2", pipeline::ACOMP_SUM2},

        {"invalid",       pipeline::ACOMP_INVALID},
        {"INVALID",       pipeline::ACOMP_INVALID},
        {"ACOMP_INVALID", pipeline::ACOMP_INVALID},
    };

    try {
        return advComponentTypeMap.at(componentTypeString);
    } catch (std::out_of_range const&){
        return pipeline::ACOMP_INVALID;
    }

    return pipeline::ACOMP_INVALID;
}

std::string pipeline::advComponentTypeToString(advanced_component_type compType){
    switch (compType){
        case ACOMP_INVALID:
            return "ACOMP_INVALID";
        case ACOMP_SUM2:
            return "ACOMP_SUM2";
    }
    return "INVALID";
}


synthesizer::settings_name synthesizer::stringToSettingName(const char*& settingNameString){
    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };
    static std::map<const char*, synthesizer::settings_name, cmp_str> settingNameMap{
        {"pitch",   synthesizer::PITCH},
        {"PITCH",   synthesizer::PITCH},

        {"attack",  synthesizer::ATTACK},
        {"ATTACK",  synthesizer::ATTACK},

        {"sustain", synthesizer::SUSTAIN},
        {"SUSTAIN", synthesizer::SUSTAIN},

        {"fade",    synthesizer::FADE},
        {"FADE",    synthesizer::FADE},

        {"fadeto",  synthesizer::FADETO},
        {"FADETO",  synthesizer::FADETO},

        {"release", synthesizer::RELEASE},
        {"RELEASE", synthesizer::RELEASE},

        {"volume",  synthesizer::VOLUME},
        {"VOLUME",  synthesizer::VOLUME},

        {"stereo",  synthesizer::STEREO},
        {"STEREO",  synthesizer::STEREO},

        {"invalid", synthesizer::INVALID},
        {"INVALID", synthesizer::INVALID},
    };

    try {
        return settingNameMap.at(settingNameString);
    } catch (std::out_of_range const&){
        return synthesizer::INVALID;
    }


    return synthesizer::INVALID;
}

synthesizer::generator_type synthesizer::stringToSynthType(const char*& synthTypeString){
    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };
    static std::map<const char*, synthesizer::generator_type, cmp_str> synthTypeMap{
        {"sine",   synthesizer::SINE},
        {"SINE",   synthesizer::SINE},

        {"square",  synthesizer::SQUARE},
        {"SQUARE",  synthesizer::SQUARE},

        {"sawtooth", synthesizer::SAWTOOTH},
        {"SAWTOOTH", synthesizer::SAWTOOTH},

        {"triangle",    synthesizer::TRIANGLE},
        {"TRIANGLE",    synthesizer::TRIANGLE},

        {"noise",  synthesizer::NOISE1},
        {"NOISE1",  synthesizer::NOISE1},

        {"invalid", synthesizer::INVALID_GEN},
        {"INVALID", synthesizer::INVALID_GEN},
    };

    try {
        return synthTypeMap.at(synthTypeString);
    } catch (std::out_of_range const&){
        return synthesizer::INVALID_GEN;
    }


    return synthesizer::INVALID_GEN;
}
