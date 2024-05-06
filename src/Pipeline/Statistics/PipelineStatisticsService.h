#ifndef PIPELINESTATISTICSSERVICE_H
#define PIPELINESTATISTICSSERVICE_H

#include "StatisticsBuffer.h"
#include "pipelineStatistics.h"
#include "../../AudioOutput/audioFormatInfo.h"

#include <chrono>
#include <fstream>

namespace statistics{
    class PipelineStatisticsService{
    public:
        PipelineStatisticsService(ulong LoopLength, uint bufferSize, const audioFormatInfo* audioFormat, uint pulseAudioLatency);
        ~PipelineStatisticsService();

        void firstInvocation();
        void loopStart();
        void loopWorkEnd();

        char record(std::string filePath, float updateTimeInterval);
        char stopRecording();

        const pipelineStatistics* getStatistics();

    private:
        void writeStatistics();

        const audioFormatInfo* audioFormat;

        const uint buffersSize;
        pipelineStatistics pStatistics;

        StatisticsBuffer<ulong> loopLengthBuffer;
        StatisticsBuffer<ulong> workLengthBuffer;

        ulong loopStartPoint;
        ulong loopWorkEndPoint;
        ulong loopEndPoint;

        ulong workLength;

        bool recording;
        std::ofstream file;
        uint sampleInterval;
        uint sampleCounter;
    };
}

#endif
