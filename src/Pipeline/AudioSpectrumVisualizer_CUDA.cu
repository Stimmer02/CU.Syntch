#include "AudioSpectrumVisualizer_CUDA.h"

__global__ void kernel_createVisualizerInputBuffer(cufftReal* workBuffer, float* bufferR, float* bufferL, uint sampleSize, uint sampleCounter, uint count){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count){
        workBuffer[sampleCounter * sampleSize + i] = bufferL[i] + bufferR[i];
    }
}


AudioSpectrumVisualizer_CUDA::AudioSpectrumVisualizer_CUDA(const audioFormatInfo* audioInfo, uint audioWindowSize, float fps): audioInfo(audioInfo){
    this->audioWindowSize = audioWindowSize;
    this->samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
    this->setFps(fps);
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    height = w.ws_row;
    width = w.ws_col;

    cudaMalloc((void**)&d_workBuffer, audioWindowSize * sizeof(cufftReal));
    cudaMalloc((void**)&d_cufftOutput, (audioWindowSize / 2 + 1) * sizeof(cufftComplex));
    cufftOutput = new cufftComplex[audioWindowSize / 2 + 1];
    bandsState = new float[width];

    cufftPlan1d(&cufftPlan, audioWindowSize, CUFFT_R2C, 1);

    highScope = 3000;
    lowScope = 20;
    volume = 24;

    running = false;
    lastWidth = 0;
}

AudioSpectrumVisualizer_CUDA::~AudioSpectrumVisualizer_CUDA(){
    delete[] bandsState;
    delete[] cufftOutput;
    cufftDestroy(cufftPlan);
    cudaFree(d_workBuffer);
    cudaFree(d_cufftOutput);
}

void AudioSpectrumVisualizer_CUDA::start(){
    readTerminalDimensions();
    sampleCounter = 0;
    for (uint i = 0; i < width; i++){
        bandsState[i] = 0;
    }

    running = true;
}

void AudioSpectrumVisualizer_CUDA::stop(){
    running = false;
    std::printf("\033[2J\033[H");
}

void AudioSpectrumVisualizer_CUDA::readTerminalDimensions(){
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

void AudioSpectrumVisualizer_CUDA::displayBuffer(pipelineAudioBuffer_CUDA* buffer){
    static const uint blockSize = 256;
    uint blockCount = (audioInfo->sampleSize + blockSize - 1) / blockSize;

    if (running == false){
        return;
    }
    if (sampleCounter < samplesPerFrame - 1){
        kernel_createVisualizerInputBuffer<<<blockCount, blockSize>>>(d_workBuffer, buffer->d_bufferR, buffer->d_bufferL, audioInfo->sampleSize, sampleCounter, audioInfo->sampleSize);
    } else if (sampleCounter == samplesPerFrame - 1){
        uint rest = audioWindowSize - (samplesPerFrame - 1) * audioInfo->sampleSize;
        kernel_createVisualizerInputBuffer<<<blockCount, blockSize>>>(d_workBuffer, buffer->d_bufferR, buffer->d_bufferL, audioInfo->sampleSize, sampleCounter, rest);

        computeFFT();
        cudaMemcpy(cufftOutput, d_cufftOutput, (audioWindowSize / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        draw("█");
    } else if (sampleCounter >= samplesPerFrame + skipSamples){
        sampleCounter = 0;
    }
    sampleCounter++;
}

float AudioSpectrumVisualizer_CUDA::setFps(float fps){
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

float AudioSpectrumVisualizer_CUDA::getFps(){
    return audioInfo->sampleRate / audioInfo->sampleSize / (samplesPerFrame + skipSamples);
}

void AudioSpectrumVisualizer_CUDA::setAudioWindowSize(uint size){
    if (size == audioWindowSize){
        return;
    }

    cufftReal* d_newWorkBuffer;
    cudaMalloc((void**)&d_newWorkBuffer, audioWindowSize * sizeof(cufftReal));
    cufftReal* d_oldWorkBuffer = d_workBuffer;

    cufftComplex* d_newCufftwOutput;
    cudaMalloc((void**)&d_newCufftwOutput, (audioWindowSize / 2 + 1) * sizeof(cufftComplex));
    cufftComplex* d_oldFftwOutput = d_cufftOutput;

    cufftComplex* newCufftOutput = new cufftComplex[audioWindowSize / 2 + 1];
    cufftComplex* oldCufftOutput = cufftOutput;

    if (audioWindowSize > size){ //to prevent segmentation fault in case of execution on multiple threads
        d_workBuffer = d_newWorkBuffer;
        d_cufftOutput = d_newCufftwOutput;
        cufftOutput = newCufftOutput;
        audioWindowSize = size;
        samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
    } else {
        audioWindowSize = size;
        samplesPerFrame = std::ceil(audioWindowSize / audioInfo->sampleSize);
        d_workBuffer = d_newWorkBuffer;
        d_cufftOutput = d_newCufftwOutput;
        cufftOutput = newCufftOutput;
    }
    cudaFree(d_oldWorkBuffer);
    cudaFree(d_oldFftwOutput);
    delete[] oldCufftOutput;
}

uint AudioSpectrumVisualizer_CUDA::getAudioWindowSize(){
    return audioWindowSize;
}

float AudioSpectrumVisualizer_CUDA::getMinFrequency(){
    return audioInfo->sampleRate / audioWindowSize;
}

float AudioSpectrumVisualizer_CUDA::getMaxFrequency(){
    return audioInfo->sampleRate / 2;
}

void AudioSpectrumVisualizer_CUDA::setVolume(float volume){
    this->volume = volume;
}

float AudioSpectrumVisualizer_CUDA::getVolume(){
    return volume;
}

float AudioSpectrumVisualizer_CUDA::setHighScope(float highScope){
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
float AudioSpectrumVisualizer_CUDA::getHighScope(){
    return highScope;
}

float AudioSpectrumVisualizer_CUDA::setLowScope(float lowScope){
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

float AudioSpectrumVisualizer_CUDA::getLowScope(){
    return lowScope;
}


void AudioSpectrumVisualizer_CUDA::draw(const char* c){
    //for better efficeincy, the following code should be executed on the GPU
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
        float& real = cufftOutput[i].x;
        float& imag = cufftOutput[i].y;
        float magnitude = sqrt(real * real + imag * imag)/audioWindowSize*volume;
        float frequency = i * audioInfo->sampleRate / audioWindowSize;

        bandsState[0] += magnitude * (1 - (currentLowBand - frequency)/bandWidth);
    }

    for (uint j = 1; j < width - 1; j++){
        float previusBandChangeIndex = bandChangeIndex;
        bandChangeIndex = std::ceil(currentHighBand / audioInfo->sampleRate * audioWindowSize);
        for (uint i = previusBandChangeIndex; i < bandChangeIndex; i++){
            float& real = cufftOutput[i].x;
            float& imag = cufftOutput[i].y;
            float magnitude = sqrt(real * real + imag * imag)/audioWindowSize*volume;
            float frequency = i * audioInfo->sampleRate / audioWindowSize;

            bandsState[j] += magnitude * (1 - (frequency - currentLowBand)/bandWidth);
            bandsState[j+1] += magnitude * (1 - (currentHighBand - frequency)/bandWidth);
        }
        currentLowBand = currentHighBand;
        currentHighBand += bandWidth;
    }

    for (uint i = bandChangeIndex; i < highScopeIndex; i++){
        float& real = cufftOutput[i].x;
        float& imag = cufftOutput[i].y;
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

void AudioSpectrumVisualizer_CUDA::computeFFT(){
    cufftExecR2C(cufftPlan, d_workBuffer, d_cufftOutput);
    cudaDeviceSynchronize();
}


