#include "PipelineStatisticsService.h"

using namespace statistics;

PipelineStatisticsService::PipelineStatisticsService(ulong loopLength, uint bufferSize, const audioFormatInfo* audioFormat, uint pulseAudioLatency): audioFormat(audioFormat), buffersSize(bufferSize), loopLengthBuffer(bufferSize), workLengthBuffer(bufferSize){
    pStatistics.loopLength = loopLength;
    pStatistics.maxLoopLength = 0;
    pStatistics.maxLoad = 0;
    pStatistics.maxWorkTime = 0;
    pStatistics.userInputLatency = 1000000.0*2 *audioFormat->sampleSize / audioFormat->sampleRate  + pulseAudioLatency;
    loopWorkEndPoint = 0;
    loopStartPoint = 0;
    loopWorkEndPoint = 0;
    loopEndPoint = 0;
    workLength = 0;

    recording = false;
    sampleCounter = 0;
    sampleInterval = 0;
}

PipelineStatisticsService::~PipelineStatisticsService(){
    if (recording){
        stopRecording();
    }
}

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
    if (recording){
        if (sampleCounter >= sampleInterval){
            writeStatistics();
            sampleCounter = 0;
        }
        sampleCounter++;
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

char PipelineStatisticsService::record(std::string filePath, float updateTimeInterval){
    if (recording){
        std::fprintf(stderr, "ERR: PipelineStatisticsService::record SERVICE ALREADY RECORDING\n");
        return 1;
    }
    file.open(filePath, std::ios::out | std::ios::trunc);
    if (!file.is_open()){
        std::fprintf(stderr, "ERR: PipelineStatisticsService::record COULD NOT OPEN FILE\n");
        return 2;
    }
    sampleInterval = updateTimeInterval * audioFormat->sampleRate / audioFormat->sampleSize;
    sampleCounter = 0;
    recording = true;
    return 0;

    file << "averageLoopLength,averageWorkTime,averageLoopLatency,averageLoad\n";
}

char PipelineStatisticsService::stopRecording(){
    if (recording == false){
        std::fprintf(stderr, "ERR: PipelineStatisticsService::stopRecording SERVICE NOT RECORDING\n");
        return 1;
    }
    file.close();
    recording = false;
    return 0;
}

void PipelineStatisticsService::writeStatistics(){
    if (recording){
        const pipelineStatistics* stats = getStatistics();
        file << stats->averageLoopLength << "," << stats->averageWorkTime << "," << stats->averageLoopLatency << "," << stats->averageLoad << "\n";
        sampleCounter++;
        if (sampleCounter == sampleInterval){
            sampleCounter = 0;
        }
    }
}