#include "AudioSpectrumVisualizer.h"

AudioSpectrumVisualizer::AudioSpectrumVisualizer(const audioFormatInfo* audioInfo, uint audioWindowSize, float fps): audioInfo(audioInfo){
    this->audioWindowSize = audioWindowSize;
    this->workBuffer = new double[audioWindowSize];
    this->samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
    this->setFps(fps);
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    height = w.ws_row;
    width = w.ws_col;
    fftwOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (audioWindowSize/2 + 1));
    bandsState = new float[width];

    fftw = fftw_plan_dft_r2c_1d(audioWindowSize, workBuffer, fftwOutput, FFTW_ESTIMATE);

    highScope = 3000;
    lowScope = 20;
    volume = 24;

    running = false;
    lastWidth = 0;
}

AudioSpectrumVisualizer::~AudioSpectrumVisualizer(){
    delete[] workBuffer;
    delete[] bandsState;
    fftw_destroy_plan(fftw);
    fftw_free(fftwOutput);
}

void AudioSpectrumVisualizer::start(){
    readTerminalDimensions();
    sampleCounter = 0;
    for (uint i = 0; i < width; i++){
        bandsState[i] = 0;
    }

    running = true;
}

void AudioSpectrumVisualizer::stop(){
    running = false;
    std::printf("\033[2J\033[H");
}

void AudioSpectrumVisualizer::readTerminalDimensions(){
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    height = w.ws_row;

    if (width == w.ws_row){
        return;
    }

    float* newBandsState = new float[w.ws_col];
    for (uint i = 0; i < w.ws_col; i++){
        newBandsState[i] = 0;
    }
    float* oldBandsState = bandsState;

    if (width < w.ws_col){ //to prevent segmentation fault in case of execution on multiple threads
        bandsState = newBandsState;
        width = w.ws_col;
    } else {
        width = w.ws_col;
        bandsState = newBandsState;
    }    
    delete[] oldBandsState;
}

void AudioSpectrumVisualizer::displayBuffer(pipelineAudioBuffer* buffer){
    if (running == false){
        return;
    }
    if (sampleCounter < samplesPerFrame - 1){
        for (uint i = 0; i < audioInfo->sampleSize; i++){
            workBuffer[sampleCounter * audioInfo->sampleSize + i] = buffer->bufferL[i] + buffer->bufferR[i];
        }
    } else if (sampleCounter == samplesPerFrame - 1){
        uint rest = audioWindowSize - (samplesPerFrame - 1) * audioInfo->sampleSize;
        for (uint i = 0; i < rest; i++){
            workBuffer[sampleCounter * audioInfo->sampleSize + i] = buffer->bufferL[i] + buffer->bufferR[i];
        }
        computeFFT();
        draw("█");
    } else if (sampleCounter >= samplesPerFrame + skipSamples){
        sampleCounter = 0;
    }
    sampleCounter++;
}

float AudioSpectrumVisualizer::setFps(float fps){
    sampleCounter = 0;
    float samplesPerSecond = audioInfo->sampleRate / audioInfo->sampleSize;
    float maxFps = samplesPerSecond / samplesPerFrame;
    if (maxFps < fps){
        skipSamples = 0;
        return maxFps;
    }

    skipSamples = std::ceil(samplesPerSecond/fps) - samplesPerFrame;
    fps = samplesPerSecond / (samplesPerFrame + skipSamples);
    return fps;
}

float AudioSpectrumVisualizer::getFps(){
    return audioInfo->sampleRate / audioInfo->sampleSize / (samplesPerFrame + skipSamples);
}

void AudioSpectrumVisualizer::setAudioWindowSize(uint size){
    if (size == audioWindowSize){
        return;
    }

    double* newWorkBuffer = new double[size];
    double* oldWorkBuffer = workBuffer;

    fftw_complex* newFftwOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (size/2 + 1));
    fftw_complex* oldFftwOutput = fftwOutput;

    if (audioWindowSize > size){ //to prevent segmentation fault in case of execution on multiple threads
        workBuffer = newWorkBuffer;
        fftwOutput = newFftwOutput;
        audioWindowSize = size;
        samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
    } else {
        audioWindowSize = size;
        samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
        workBuffer = newWorkBuffer;
        fftwOutput = newFftwOutput;
    }
    delete[] oldWorkBuffer;
    fftw_free(oldFftwOutput);
}

uint AudioSpectrumVisualizer::getAudioWindowSize(){
    return audioWindowSize;
}

float AudioSpectrumVisualizer::getMinFrequency(){
    return audioInfo->sampleRate / audioWindowSize;
}

float AudioSpectrumVisualizer::getMaxFrequency(){
    return audioInfo->sampleRate / 2;
}

void AudioSpectrumVisualizer::setVolume(float volume){
    this->volume = volume;
}

float AudioSpectrumVisualizer::getVolume(){
    return volume;
}

float AudioSpectrumVisualizer::setHighScope(float highScope){
    if (highScope < lowScope){
        return this->highScope;
    }
    if (highScope > getMaxFrequency()){
        highScope = getMaxFrequency();
        return highScope;
    }
    this->highScope = highScope;
    return highScope;
}
float AudioSpectrumVisualizer::getHighScope(){
    return highScope;
}

float AudioSpectrumVisualizer::setLowScope(float lowScope){
    if (lowScope > highScope){
        return this->lowScope;
    }
    if (lowScope < getMinFrequency()){
        lowScope = getMinFrequency();
        return lowScope;
    }
    this->lowScope = lowScope;
    return lowScope;
}

float AudioSpectrumVisualizer::getLowScope(){
    return lowScope;
}

void AudioSpectrumVisualizer::draw(const char* c){
    uint lowScopeIndex = std::ceil(lowScope / audioInfo->sampleRate * audioWindowSize);
    uint highScopeIndex = std::ceil(highScope / audioInfo->sampleRate * audioWindowSize);

    float bandWidth = (highScope - lowScope) / width;

    float currentLowBand = lowScope + bandWidth / 2;
    float currentHighBand = currentLowBand + bandWidth; 

    uint bandChangeIndex = std::ceil(currentLowBand / audioInfo->sampleRate * audioWindowSize);

    for (uint i = 0; i < width; i++){
        bandsState[i] = 0;
    }

    for (uint i = lowScopeIndex; i < bandChangeIndex; i++){
        double& real = fftwOutput[i][0];
        double& imag = fftwOutput[i][1];
        float magnitude = sqrt(real * real + imag * imag)/audioWindowSize*volume;
        float frequency = i * audioInfo->sampleRate / audioWindowSize;

        bandsState[0] += magnitude * (1 - (currentLowBand - frequency)/bandWidth);
    }

    for (uint j = 1; j < width - 1; j++){
        float previusBandChangeIndex = bandChangeIndex;
        bandChangeIndex = std::ceil(currentHighBand / audioInfo->sampleRate * audioWindowSize);
        for (uint i = previusBandChangeIndex; i < bandChangeIndex; i++){
            double& real = fftwOutput[i][0];
            double& imag = fftwOutput[i][1];
            float magnitude = sqrt(real * real + imag * imag)/audioWindowSize*volume;
            float frequency = i * audioInfo->sampleRate / audioWindowSize;

            bandsState[j] += magnitude * (1 - (frequency - currentLowBand)/bandWidth);
            bandsState[j+1] += magnitude * (1 - (currentHighBand - frequency)/bandWidth);
        }
        currentLowBand = currentHighBand;
        currentHighBand += bandWidth;
    }

    for (uint i = bandChangeIndex; i < highScopeIndex; i++){
        double& real = fftwOutput[i][0];
        double& imag = fftwOutput[i][1];
        float magnitude = sqrt(real * real + imag * imag)/audioWindowSize*volume;
        float frequency = i * audioInfo->sampleRate / audioWindowSize;

        bandsState[width - 1] += magnitude * (1 - (frequency - currentLowBand)/bandWidth);
    }

    std::string out = "\033[H";
    for (uint i = 0; i < height - 1; i++){
        float threshold = float(height - 1 - i) / (height - 1);
        for (uint j = 0; j < width; j++){
            if (std::log1p(bandsState[j] * 9) / std::log1p(10) >= threshold){
                out += c;
            } else {
                out += " ";
            }
        }
        out += "\n";
    }

    if (width < 18){
       std::printf("%s%4dHz %5dHz", out.c_str(), int(lowScope), int(highScope)); 
    } else {
        if (lastWidth != width){
            bottomLine = std::string(width - 18, '-');
            lastWidth = width;
        }
        std::printf("%s%4dHz <%s> %5dHz", out.c_str(), int(lowScope), bottomLine.c_str(), int(highScope));
    }
}

void AudioSpectrumVisualizer::computeFFT(){
    fftw_execute(fftw);
}


