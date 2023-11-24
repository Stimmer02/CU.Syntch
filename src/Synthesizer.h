#ifndef _SYNTHESIZER_H
#define _SYNTHESIZER_H

#include "Synthesizer/noteBuffer.h"
#include "Synthesizer/settings.h"
#include "Synthesizer/IGenerator.h"
#include "Synthesizer/Generator_Sine.h"
#include "Synthesizer/Generator_Square.h"
#include "Synthesizer/Generator_Triangle.h"
#include "UserInput/KeyboardDoubleBuffer.h"
#include "Pipeline/pipelineAudioBuffer.h"
#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/keyboardTransferBuffer.h"

// #include <mad.h>

#define PI 3.1415926595


namespace synthesizer {
    class Synthesizer{
    public:
        Synthesizer(const audioFormatInfo& audioFormatInfo, const ushort& keyCount);
        ~Synthesizer();
        void generateSample(pipelineAudioBuffer* audioBuffer, const keyboardTransferBuffer* keyboardState);
        struct settings* getSettings();
        void setGenerator(generator_type type);
        char setPitch(const char& value, const bool& add = false);

    private:
        void calculateFrequencies();

        IGenerator* soundGenerator;
        noteBuffer* notes;
        synthesizer::settings settings;
    };

    enum settings_name {
        PITCH,
        ATTACK,
        SUSTAIN,
        FADE,
        RELEASE,
        VOLUME
    };
}
#endif
