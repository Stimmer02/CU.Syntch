#include "PipelineStatisticsService.h"

using namespace statistics;

PipelineStatisticsService::PipelineStatisticsService(ulong loopLength, uint bufferSize, audioFormatInfo audioFormat, uint pulseAudioLatency) : buffersSize(bufferSize), loopLengthBuffer(bufferSize), workLengthBuffer(bufferSize){
    pStatistics.loopLength = loopLength;
    pStatistics.maxLoopLength = 0;
    pStatistics.maxLoad = 0;
    pStatistics.maxWorkTime = 0;
    pStatistics.userInputLatency = 1000000.0*2 *audioFormat.sampleSize / audioFormat.sampleRate  + pulseAudioLatency;
    loopWorkEndPoint = 0;
    loopStartPoint = 0;
    loopLength = 0;
    loopWorkEndPoint = 0;
}

PipelineStatisticsService::~PipelineStatisticsService(){}

void PipelineStatisticsService::loopStart(){
    loopEndPoint = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    ulong loopLength = loopEndPoint - loopStartPoint;
    loopLengthBuffer.push(loopLength);
    if (loopLength > pStatistics.maxLoopLength){
        pStatistics.maxLoopLength = loopLength;
    }

    double load = 100.0 * workLength / pStatistics.loopLength;
    if (load > pStatistics.maxLoad){
        pStatistics.maxLoad = load;
    }

    loopStartPoint = loopEndPoint;
}

void PipelineStatisticsService::loopWorkEnd(){
    loopWorkEndPoint = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    workLength = loopWorkEndPoint - loopStartPoint;
    workLengthBuffer.push(workLength);
    if (workLength > pStatistics.maxWorkTime){
        pStatistics.maxWorkTime = workLength;
    }
}

void PipelineStatisticsService::firstInvocation(){
    loopStartPoint = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    loopWorkEndPoint = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
    workLength = 0;
}

const pipelineStatistics* PipelineStatisticsService::getStatistics(){
    pStatistics.averageLoopLength = loopLengthBuffer.average();
    pStatistics.averageWorkTime = workLengthBuffer.average();
    pStatistics.averageLoopLatency = pStatistics.averageLoopLength - pStatistics.loopLength;
    pStatistics.averageLoad = 100.0 * pStatistics.averageWorkTime / pStatistics.loopLength;
    return &pStatistics;
}
