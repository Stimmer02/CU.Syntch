#ifndef _AUDIOPIPELINESUBSTITUTE_H
#define _AUDIOPIPELINESUBSTITUTE_H


#include "AudioOutput/audioFormatInfo.h"
#include "Synthesizer/settings.h"
#include "UserInput/AKeyboardRecorder.h"
#include "UserInput/KeyboardRecorder_DevInput.h"
#include "UserInput/keyboardTransferBuffer.h"
#include "AudioOutput/IOutStream.h"
#include "AudioOutput/audioBuffer.h"
#include "AudioOutput/OutStream_PulseAudio.h"
#include "Pipeline/pipelineAudioBuffer.h"
#include "Synthesizer.h"
#include "Pipeline/BufferConverter/IBufferConverter.h"
#include "Pipeline/BufferConverter/BufferConverter_Mono8.h"
#include "Pipeline/BufferConverter/BufferConverter_Mono16.h"
#include "Pipeline/BufferConverter/BufferConverter_Mono24.h"
#include "Pipeline/BufferConverter/BufferConverter_Mono32.h"
#include "Pipeline/Statistics/PipelineStatisticsService.h"
#include "Pipeline/Statistics/pipelineStatistics.h"
#include "AudioOutput/AudioRecorder.h"


class AudioPipelineSubstitute{
public:
    AudioPipelineSubstitute(audioFormatInfo audioInfo, ushort keyCount, AKeyboardRecorder* midiInput);
    ~AudioPipelineSubstitute();

    void start();
    void stop();

    void startRecording();
    void stopRecording();
    bool isRecording();

    const statistics::pipelineStatistics* getStatistics();

    // void addSynth();

    const audioFormatInfo* getAudioInfo();

    const synthesizer::settings* getSynthSettings(ushort id);

    // template<typename T>
    void setSynthSettings(ushort id, synthesizer::settings_name settingsName, double value);

private:
    void pipelineThreadFunction();

    audioFormatInfo audioInfo;
    ushort keyCount;
    AKeyboardRecorder* midiInput;
    statistics::PipelineStatisticsService* statisticsService;

    keyboardTransferBuffer* keyboardState;
    IOutStream* audioOutput;
    synthesizer::Synthesizer* synth;

    pipelineAudioBuffer* pipelineBuffer;
    audioBuffer* buffer;
    IBufferConverter* bufferConverter;
    AudioRecorder audioRecorder;

    bool running;
    bool recording;
    std::thread* pipelineThread;
};

#endif
