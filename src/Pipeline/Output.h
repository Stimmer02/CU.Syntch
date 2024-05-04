#ifndef PIPELINEOUTPUT_H
#define PIPELINEOUTPUT_H

#include "../AudioOutput/AudioRecorder.h"
#include "../AudioOutput/IOutStream.h"
#include "../AudioOutput/OutStream_PulseAudio.h"
#include "BufferConverter/IBufferConverter_CUDA.h"
#include "BufferConverter/IBufferConverter_CUDA.h"
#include "BufferConverter/BufferConverter_Mono8_CUDA.h"
#include "BufferConverter/BufferConverter_Mono16_CUDA.h"
#include "BufferConverter/BufferConverter_Mono24_CUDA.h"
#include "BufferConverter/BufferConverter_Mono32_CUDA.h"
#include "BufferConverter/BufferConverter_Stereo8_CUDA.h"
#include "BufferConverter/BufferConverter_Stereo16_CUDA.h"
#include "BufferConverter/BufferConverter_Stereo24_CUDA.h"
#include "BufferConverter/BufferConverter_Stereo32_CUDA.h"

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
