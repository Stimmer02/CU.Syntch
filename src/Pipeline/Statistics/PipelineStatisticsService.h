#ifndef PIPELINESTATISTICSSERVICE_H
#define PIPELINESTATISTICSSERVICE_H

#include "StatisticsBuffer.h"
#include "pipelineStatistics.h"
#include "../../AudioOutput/audioFormatInfo.h"

#include <chrono>

namespace statistics{
    class PipelineStatisticsService{
    public:
        PipelineStatisticsService(ulong LoopLength, uint bufferSize, audioFormatInfo audioFormat, uint pulseAudioLatency);
        ~PipelineStatisticsService();

        void firstInvocation();
        void loopStart();
        void loopWorkEnd();

        const pipelineStatistics* getStatistics();

    private:
        const uint buffersSize;
        pipelineStatistics pStatistics;

        StatisticsBuffer<ulong> loopLengthBuffer;
        StatisticsBuffer<ulong> workLengthBuffer;

        ulong loopStartPoint;
        ulong loopWorkEndPoint;
        ulong loopEndPoint;

        ulong workLength;
    };
}

#endif
