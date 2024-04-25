#ifndef SYNTHESIZER_CUDA_H
#define SYNTHESIZER_CUDA_H

#include "Synthesizer/DynamicsController_CUDA.h"
#include "Synthesizer/NoteBufferHandler_CUDA.h"
#include "Synthesizer/settings_CUDA.h"
#include "Synthesizer/Generator_CUDA.h"
#include "UserInput/KeyboardDoubleBuffer.h"
#include "Pipeline/pipelineAudioBuffer_CUDA.h"
#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/keyboardTransferBuffer_CUDA.h"

#include <fstream>

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
        STEREO,
        INVALID
    };

    class Synthesizer_CUDA{
    public:
        Synthesizer_CUDA(const audioFormatInfo& audioFormatInfo, const ushort& keyCount);
        ~Synthesizer_CUDA();

        void generateSample(pipelineAudioBuffer_CUDA* audioBuffer, const keyboardTransferBuffer_CUDA* keyboardState);

        const struct settings_CUDA* getSettings();
        void setSettings(const settings_name& settingsName, const float& value);
        void setGenerator(const generator_type& type);
        generator_type getGeneratorType();

        char saveConfig(std::string path);
        char loadConfig(std::string path);

    private:
        void calculateFrequencies();
        void calculateStereoFactor();
        void mixAudio(pipelineAudioBuffer_CUDA*& audioBuffer);

        Generator_CUDA* soundGenerator;
        NoteBufferHandler_CUDA notes;
        DynamicsController_CUDA dynamicsController;
        synthesizer::settings_CUDA settings;
        synthesizer::settings_CUDA* d_settings;
    };
}
#endif
