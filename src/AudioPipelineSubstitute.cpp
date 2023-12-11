#include "AudioPipelineSubstitute.h"

AudioPipelineSubstitute::AudioPipelineSubstitute(audioFormatInfo audioInfo, ushort keyCount, AKeyboardRecorder* midiInput){
    this->audioInfo = audioInfo;
    this->keyCount = keyCount;
    this->midiInput = midiInput;

    keyboardState = new keyboardTransferBuffer(audioInfo.sampleSize, keyCount);
    audioOutput = new OutStream_PulseAudio();
    synth = new synthesizer::Synthesizer(audioInfo, keyCount);

    pipelineBuffer = new pipelineAudioBuffer(audioInfo.sampleSize);
    buffer = new audioBuffer(audioInfo.sampleSize*audioInfo.channels*audioInfo.bitDepth/8);
    buffer->count = buffer->size;

    running = false;
    recording = false;
    pipelineThread = nullptr;


    if (audioInfo.channels == 1){
        if (audioInfo.bitDepth <= 8){
            bufferConverter = new BufferConverter_Mono8();
        } else if (audioInfo.bitDepth <= 16){
            bufferConverter = new BufferConverter_Mono16();
        } else if (audioInfo.bitDepth <= 24){
            bufferConverter = new BufferConverter_Mono24();
        } else if (audioInfo.bitDepth <= 32){
            bufferConverter = new BufferConverter_Mono32();
        } else {
            bufferConverter = nullptr;
            std::fprintf(stderr, "ERR AudioPipelineSubstitute::AudioPipelineSubstitute: UNSUPPORTED BIT DEPTH\n");
            exit(1);
        }
    } else if (audioInfo.channels == 2){
        if (audioInfo.bitDepth <= 8){
            bufferConverter = new BufferConverter_Stereo8();
        } else if (audioInfo.bitDepth <= 16){
            bufferConverter = new BufferConverter_Stereo16();
        } else if (audioInfo.bitDepth <= 24){
            bufferConverter = new BufferConverter_Stereo24();
        } else if (audioInfo.bitDepth <= 32){
            bufferConverter = new BufferConverter_Stereo32();
        } else {
            bufferConverter = nullptr;
            std::fprintf(stderr, "ERR AudioPipelineSubstitute::AudioPipelineSubstitute: UNSUPPORTED BIT DEPTH\n");
            exit(1);
        }
    } else {
        bufferConverter = nullptr;
        std::fprintf(stderr, "ERR AudioPipelineSubstitute::AudioPipelineSubstitute: UNSUPPORTED CHANNEL COUNT\n");
        exit(1);
    }
    statisticsService = new statistics::PipelineStatisticsService(audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate, 64, audioInfo, 0);
}

AudioPipelineSubstitute::~AudioPipelineSubstitute(){
    stop();

    if (bufferConverter != nullptr){
        delete bufferConverter;
    }

    delete statisticsService;
    delete keyboardState;
    delete audioOutput;
    delete synth;
    delete pipelineBuffer;
    delete buffer;
}

void AudioPipelineSubstitute::start(){
    if (running) return;
    if (audioOutput->init(audioInfo, "Synth", "Synthesizer")) return;
    if (midiInput->start()) return;
    while (midiInput->isRunning() == false);

    if (pipelineThread != nullptr){
        delete pipelineThread;
    }
    pipelineThread = new std::thread(&AudioPipelineSubstitute::pipelineThreadFunction, this);

    running = true;
}

void AudioPipelineSubstitute::stop(){
    if (running == false){
        return;
    }
    running = false;
    stopRecording();
    if (pipelineThread->joinable()){
        pipelineThread->join();
    }
    midiInput->stop();
}

void AudioPipelineSubstitute::startRecording(){
    if (recording) {
        return;
    }
    if (audioRecorder.init(audioInfo)){
        return;
    }
    recording = true;
}

void AudioPipelineSubstitute::startRecording(std::string outPath){
    if (recording) {
        return;
    }
    if (audioRecorder.init(audioInfo, outPath)){
        return;
    }
    recording = true;
}

void AudioPipelineSubstitute::stopRecording(){
    if (recording == false) {
        return;
    }
    recording = false;
    audioRecorder.closeFile();
}

bool AudioPipelineSubstitute::isRecording(){
    return recording;
}

void printLastBuffer(const float* buffer, const uint& size){
    static FILE* dumpFile = fopen("streamDump.txt", "w");
    static char space1[101] = "                                                                                                   ";
    static char space2[101] = "                                                                                                   ";

    for (uint i = 0; i < size; i++){
        uint fill = (buffer[i]+1)/2*100;
        space1[fill] = 0x0;
        space2[100 - fill] = 0x0;
        std::fprintf(dumpFile, "%+1.7f |%s*%s|\n", buffer[i], space1, space2);
        space1[fill] = ' ';
        space2[100 - fill] = ' ';
    }
    std::fprintf(dumpFile, "-------------------------BUFF_END-------------------------\n");
}



void AudioPipelineSubstitute::pipelineThreadFunction(){
    ulong sampleTimeLength = audioInfo.sampleSize*long(1000000)/audioInfo.sampleRate;
    midiInput->buffer->swapActiveBuffer();
    ulong nextLoop = midiInput->buffer->getActivationTimestamp() + sampleTimeLength;
    statisticsService->firstInvocation();

#ifdef _WIN32
    *((unsigned int*)0XD) = 0xDEAD;
#endif

    while (running){
        std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>(std::chrono::nanoseconds((nextLoop)*1000)));
        statisticsService->loopStart();
        midiInput->buffer->swapActiveBuffer();


        nextLoop += sampleTimeLength;
        keyboardState->convertBuffer(midiInput->buffer);
        midiInput->buffer->clearInactiveBuffer();


        synth->generateSample(pipelineBuffer, keyboardState);
        printLastBuffer(pipelineBuffer->bufferL, pipelineBuffer->size);
        bufferConverter->toPCM(pipelineBuffer, buffer);
        statisticsService->loopWorkEnd();
        audioOutput->playBuffer(buffer);
        if (recording){
            audioRecorder.saveBuffer(buffer);
        }
    }
}

void AudioPipelineSubstitute::recordUntilStreamEmpty(MIDI::MidiFileReader& midi, std::string filename){
    midi.rewindFile();
    if (filename.empty()){
        startRecording();
    } else {
        startRecording(filename);
    }

    while (midi.isFileReady() && !midi.eofChunk(0)){
        midi.fillBuffer(keyboardState, 0);
        synth->generateSample(pipelineBuffer, keyboardState);
        bufferConverter->toPCM(pipelineBuffer, buffer);
        audioRecorder.saveBuffer(buffer);
    }

    for (uint i = 0; i <= 2*audioInfo.sampleRate/audioInfo.sampleSize; i++){
        synth->generateSample(pipelineBuffer, keyboardState);
        bufferConverter->toPCM(pipelineBuffer, buffer);
        audioRecorder.saveBuffer(buffer);
    }
    stopRecording();
}



const statistics::pipelineStatistics* AudioPipelineSubstitute::getStatistics(){
    return statisticsService->getStatistics();
}

const audioFormatInfo* AudioPipelineSubstitute::getAudioInfo(){
    return &audioInfo;
}

const synthesizer::settings* AudioPipelineSubstitute::getSynthSettings(const ushort& id){
    return synth->getSettings();
}

synthesizer::generator_type AudioPipelineSubstitute::getSynthType(const ushort& id){
    return synth->getGeneratorType();
}

void AudioPipelineSubstitute::setSynthSettings(const ushort& id, const synthesizer::settings_name& settingsName, const float& value){
    synth->setSettings(settingsName, value);
}

void AudioPipelineSubstitute::setSynthSettings(const ushort& id, const synthesizer::generator_type& type){
    synth->setGenerator(type);
}

char AudioPipelineSubstitute::saveSynthConfig(std::string path, ushort id){
    return synth->saveConfig(path);
}

char AudioPipelineSubstitute::loadSynthConfig(std::string path, ushort id){
    return synth->loadConfig(path);
}



