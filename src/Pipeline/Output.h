#ifndef PIPELINEOUTPUT_H
#define PIPELINEOUTPUT_H

#include "../AudioOutput/AudioRecorder.h"
#include "../AudioOutput/IOutStream.h"
#include "../AudioOutput/OutStream_PulseAudio.h"
#include "BufferConverter/IBufferConverter.h"
#include "BufferConverter/IBufferConverter.h"
#include "BufferConverter/BufferConverter_Mono8.h"
#include "BufferConverter/BufferConverter_Mono16.h"
#include "BufferConverter/BufferConverter_Mono24.h"
#include "BufferConverter/BufferConverter_Mono32.h"
#include "BufferConverter/BufferConverter_Stereo8.h"
#include "BufferConverter/BufferConverter_Stereo16.h"
#include "BufferConverter/BufferConverter_Stereo24.h"
#include "BufferConverter/BufferConverter_Stereo32.h"

#include <chrono>


namespace pipeline{
    class Output{
    public:
        Output();
        ~Output();

        char init(audioFormatInfo audioInfo);
        bool isReady();

        void play(pipelineAudioBuffer_CUDA* pipelineBuffer);
        void onlyRecord(pipelineAudioBuffer_CUDA* pipelineBuffer);
        void onlyRecord(pipelineAudioBuffer_CUDA* pipelineBuffer, std::chrono::_V2::system_clock::time_point& timeEnd);

        char startRecording();
        char startRecording(std::string outPath);
        char stopRecording();
        bool isRecording();

    private:
        void celanup();

        audioFormatInfo audioInfo;

        audioBuffer* buffer;
        IBufferConverter_CUDA* bufferConverter;
        AudioRecorder audioRecorder;
        IOutStream* audioOutput;

        bool recording;
    };
}

#endif
