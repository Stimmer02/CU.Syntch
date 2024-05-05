#include "Output.h"

using namespace pipeline;

Output::Output(){
    audioOutput = new OutStream_PulseAudio();
    bufferConverter = nullptr;
    buffer = nullptr;

    recording = false;
}

Output::~Output(){
    delete audioOutput;
    stopRecording();
    celanup();
}

void Output::celanup(){
    if (bufferConverter != nullptr){
        delete bufferConverter;
        bufferConverter = nullptr;
    }
    if (buffer != nullptr){
        delete buffer;
        buffer = nullptr;
    }
}

char Output::init(audioFormatInfo audioInfo){
    this->audioInfo = audioInfo;
    celanup();

    if (audioOutput->init(audioInfo, "Synth", "Synthesizer_CUDA")){
        return 1;
    }

    buffer = new audioBuffer(audioInfo.sampleSize * audioInfo.channels * audioInfo.bitDepth/8);
    buffer->count = buffer->size;

    if (audioInfo.channels == 1){
        if (audioInfo.bitDepth <= 8){
            bufferConverter = new BufferConverter_Mono8_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 16){
            bufferConverter = new BufferConverter_Mono16_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 24){
            bufferConverter = new BufferConverter_Mono24_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 32){
            bufferConverter = new BufferConverter_Mono32_CUDA(audioInfo.sampleSize);
        } else {
            bufferConverter = nullptr;
            std::fprintf(stderr, "ERR pipeline::Output::init: UNSUPPORTED BIT DEPTH\n");
            return 2;
        }
    } else if (audioInfo.channels == 2){
        if (audioInfo.bitDepth <= 8){
            bufferConverter = new BufferConverter_Stereo8_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 16){
            bufferConverter = new BufferConverter_Stereo16_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 24){
            bufferConverter = new BufferConverter_Stereo24_CUDA(audioInfo.sampleSize);
        } else if (audioInfo.bitDepth <= 32){
            bufferConverter = new BufferConverter_Stereo32_CUDA(audioInfo.sampleSize);
        } else {
            bufferConverter = nullptr;
            std::fprintf(stderr, "ERR pipeline::Output::init: UNSUPPORTED BIT DEPTH\n");
            return 3;
        }
    } else {
        bufferConverter = nullptr;
        std::fprintf(stderr, "ERR pipeline::Output::init: UNSUPPORTED CHANNEL COUNT\n");
        return 4;
    }

    return 0;
}

//INITIALIZE BEFORE USING THIS!
void Output::play(pipelineAudioBuffer_CUDA* pipelineBuffer){
    try {
        bufferConverter->toPCM(pipelineBuffer, buffer);
        
        audioOutput->playBuffer(buffer);
        if (recording){
            audioRecorder.saveBuffer(buffer);
        }
    } catch (std::exception& e){
        std::fprintf(stderr, "ERR pipeline::Output::play: %s\n", e.what());
    }
}

void Output::onlyRecord(pipelineAudioBuffer_CUDA* pipelineBuffer){
    try {
        bufferConverter->toPCM(pipelineBuffer, buffer);
        audioRecorder.saveBuffer(buffer);
    } catch (std::exception& e){
        std::fprintf(stderr, "ERR pipeline::Output::play: %s\n", e.what());
    }
}

void Output::onlyRecord(pipelineAudioBuffer_CUDA* pipelineBuffer, std::chrono::_V2::system_clock::time_point& timeEnd){
    try {
        bufferConverter->toPCM(pipelineBuffer, buffer);
        timeEnd = std::chrono::system_clock::now();
        audioRecorder.saveBuffer(buffer);
    } catch (std::exception& e){
        std::fprintf(stderr, "ERR pipeline::Output::play: %s\n", e.what());
    }
}

char Output::startRecording(){
    if (recording) {
        return 1;
    }
    if (audioRecorder.init(audioInfo)){
        return 2;
    }
    recording = true;

    return 0;
}

char Output::startRecording(std::string outPath){
    if (recording) {
        return 1;
    }
    if (audioRecorder.init(audioInfo, outPath)){
        return 2;
    }
    recording = true;

    return 0;
}

char Output::stopRecording(){
    if (recording == false) {
        return 1;
    }
    recording = false;
    audioRecorder.closeFile();

    return 0;
}

bool Output::isRecording(){
    return recording;
}
bool Output::isReady(){
    return bufferConverter != nullptr;
}
