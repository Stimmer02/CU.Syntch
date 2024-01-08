#ifndef AUDIOPIPELINEMANAGER_H
#define AUDIOPIPELINEMANAGER_H

#include "Pipeline/Input.h"
#include "Pipeline/Output.h"
#include "Pipeline/Statistics/PipelineStatisticsService.h"
#include "Pipeline/Statistics/pipelineStatistics.h"

#include "Pipeline/pipelineAudioBuffer.h"
#include "UserInput/MIDI/MidiFileReader.h"


namespace pipeline{
    class AudioPipelineManager{
    public:
        AudioPipelineManager(audioFormatInfo audioInfo, ushort keyCount);
        ~AudioPipelineManager();

        char start();
        void stop();

        const statistics::pipelineStatistics* getStatistics();
        const audioFormatInfo* getAudioInfo();

        void recordUntilStreamEmpty(MIDI::MidiFileReader& midi, std::string filename = "");


        //OUTPUT CONTROL
        void startRecording();
        void startRecording(std::string outPath);
        void stopRecording();
        bool isRecording();


        //INPUT CONTROL
        void pauseInput();
        void reausumeInput();

        short addInput(AKeyboardRecorder* input);
        char removeInput(short ID);
        short getInputCount();

        short addSynthesizer();
        char removeSynthesizer(short ID);
        short getSynthesizerCount();

        char connectInputToSynth(short inputID, short synthID);

        const synthesizer::settings* getSynthSettings(const ushort& id);
        synthesizer::generator_type getSynthType(const ushort& id);
        void setSynthSettings(const ushort& id, const synthesizer::settings_name& settingsName, const float& value);
        void setSynthSettings(const ushort& id, const synthesizer::generator_type& type);

        char saveSynthConfig(std::string path, ushort id);
        char loadSynthConfig(std::string path, ushort id);

    private:
        void pipelineThreadFunction();

        const audioFormatInfo audioInfo;
        const ushort keyCount;

        Input input;
        Output output;

        pipelineAudioBuffer* temporaryBuffer; //TODO:remove this

        statistics::PipelineStatisticsService* statisticsService;

        bool running;
        std::thread* pipelineThread;
    };
}

#endif
