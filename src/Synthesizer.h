#ifndef _SYNTHESIZER_H
#define _SYNTHESIZER_H

#include "Synthesizer/DynamicsController.h"
#include "Synthesizer/noteBuffer.h"
#include "Synthesizer/settings.h"
#include "Synthesizer/IGenerator.h"
#include "Synthesizer/Generator_Sine.h"
#include "Synthesizer/Generator_Square.h"
#include "Synthesizer/Generator_Sawtooth.h"
#include "UserInput/KeyboardDoubleBuffer.h"
#include "Pipeline/pipelineAudioBuffer.h"
#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/keyboardTransferBuffer.h"

// #include <mad.h>

#define PI 3.1415926595


namespace synthesizer {
    enum settings_name {
        PITCH,
        ATTACK,
        SUSTAIN,
        FADE,
        FADETO,
        RELEASE,
        VOLUME,
        STEREO
    };

    class Synthesizer{
    public:
        Synthesizer(const audioFormatInfo& audioFormatInfo, const ushort& keyCount);
        ~Synthesizer();

        void generateSample(pipelineAudioBuffer* audioBuffer, const keyboardTransferBuffer* keyboardState);

        const struct settings* getSettings();
        void setSettings(const settings_name& settingsName, const double& value);
        void setGenerator(const generator_type& type);
        generator_type getGeneratorType();

    private:
        void calculateFrequencies();
        void calculateStereoFactor();
        void mixAudio(pipelineAudioBuffer*& audioBuffer);

        IGenerator* soundGenerator;
        generator_type generatorType;
        noteBuffer* notes;
        DynamicsController dynamicsController;
        synthesizer::settings settings;
    };
}
#endif
