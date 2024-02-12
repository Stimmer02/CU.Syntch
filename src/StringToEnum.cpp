#include "StringToEnum.h"

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
