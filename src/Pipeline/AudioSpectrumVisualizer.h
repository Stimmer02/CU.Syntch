#ifndef AUDIOSPECTRUMVISUALIZER_H
#define AUDIOSPECTRUMVISUALIZER_H

#include "./pipelineAudioBuffer.h"
#include "../AudioOutput/audioFormatInfo.h"

#include <fftw3.h>
#include <cmath>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>

class AudioSpectrumVisualizer{
public:
    AudioSpectrumVisualizer(const audioFormatInfo* audioInfo, uint audioWindowSize, float fps);
    ~AudioSpectrumVisualizer();

    void start();
    void stop();
    void readTerminalDimensions();
    void displayBuffer(pipelineAudioBuffer* buffer);
    float setFps(float fps);
    float getFps();
    void setAudioWindowSize(uint size);
    uint getAudioWindowSize();

    float getMinFrequency();
    float getMaxFrequency();

    void setVolume(float volume);
    float getVolume();
    float setHighScope(float highScope);
    float getHighScope();
    float setLowScope(float lowScope);
    float getLowScope();

private:
    void draw(const char* c = "#");
    void computeFFT();
    
    bool running;
    uint width;
    uint height;

    const audioFormatInfo* audioInfo;
    float* bandsState;
    uint skipSamples;
    uint sampleCounter;

    uint audioWindowSize;
    uint samplesPerFrame; //how many times displayBuffer() has to be called to execute computeFFT()
    double* workBuffer;
    fftw_complex* fftwOutput;
    fftw_plan fftw;

    float highScope;
    float lowScope;

    float volume;

    uint lastWidth;
    std::string bottomLine;
};
#endif
