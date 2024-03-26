#include "KeyboardRecorder_MidiFile.h"
#include "KeyboardDoubleBuffer_MidiFile.h"

using namespace MIDI;

KeyboardRecorder_MidiFile::KeyboardRecorder_MidiFile(): emptyBuffer(){
    running = false;
    playing = false;

    sampleSize = 0;
    sampleRate = 0;

    path = "";

    emptyBuffer = nullptr;
    midiBuffer = nullptr;
    buffer = nullptr;
    playCounter = nullptr;
}

KeyboardRecorder_MidiFile::~KeyboardRecorder_MidiFile(){
    if (emptyBuffer != nullptr){
        delete emptyBuffer;
    }

    if (midiBuffer != nullptr){
        delete midiBuffer;
    }
}

char KeyboardRecorder_MidiFile::init(std::string path, const uint& sampleSize, const uint& sampleRate){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::init CANNOT INITAILIZE WHILE RUNNING FLAG IS TRUE\n");
        return -1;
    }

    if (emptyBuffer != nullptr){
        delete emptyBuffer;
    }
    if (midiBuffer != nullptr){
        delete midiBuffer;
    }

    this->sampleSize = sampleSize;
    this->sampleRate = sampleRate;

    emptyBuffer = new KeyboardDoubleBuffer_Empty(sampleSize, 127);
    midiBuffer = new KeyboardDoubleBuffer_MidiFile(path, sampleSize, sampleRate);
    pause();

    if (midiBuffer->fileReader.isFileReady() == false){
        delete midiBuffer;
        midiBuffer = nullptr;
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::init FILE \"%s\" DOES NOT EXIST OR USER DOES NOT HAVE PERMISSIONS TO READ IT\n", path.c_str());
        return -2;
    }

    this->path = path;
    return 0;
}

char KeyboardRecorder_MidiFile::init(const uint& sampleSize, const uint& sampleRate){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::init CANNOT INITAILIZE WHILE RUNNING FLAG IS TRUE\n");
        return -1;
    }

    if (emptyBuffer != nullptr){
        delete emptyBuffer;
    }
    if (midiBuffer != nullptr){
        delete midiBuffer;
        midiBuffer = nullptr;
    }

    this->sampleSize = sampleSize;
    this->sampleRate = sampleRate;

    emptyBuffer = new KeyboardDoubleBuffer_Empty(sampleSize, 127);
    this->buffer = emptyBuffer;

    return 0;
}

char KeyboardRecorder_MidiFile::reInit(const uint& sampleSize, const uint& sampleRate){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::reInit CANNOT RE-INITAILIZE WHILE WHILE RUNNING FLAG IS TRUE\n");
        return -1;
    }

    if (emptyBuffer == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::reInit CANNOT RE-INITAILIZE WITHOUT INITIALIZATION\n");
        return -2;
    }

    delete emptyBuffer;

    this->sampleSize = sampleSize;
    this->sampleRate = sampleRate;

    emptyBuffer = new KeyboardDoubleBuffer_Empty(sampleSize, 127);
    pause();

    if (midiBuffer != nullptr){
        delete midiBuffer;
        midiBuffer = new KeyboardDoubleBuffer_MidiFile(path, sampleSize, sampleRate);
    }

    return 0;
}

char KeyboardRecorder_MidiFile::reInitFile(std::string path){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::reInitFile CANNOT RE-INITAILIZE WHILE WHILE RUNNING FLAG IS TRUE\n");
        return -1;
    }

    if (emptyBuffer == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::reInitFile CANNOT RE-INITAILIZE WITHOUT INITIALIZATION\n");
        return -2;
    
    }

    if (playing){
        pause();
    }

    if (midiBuffer != nullptr){
        delete midiBuffer;
    }

    midiBuffer = new KeyboardDoubleBuffer_MidiFile(path, sampleSize, sampleRate);

    if (midiBuffer->fileReader.isFileReady() == false){
        delete midiBuffer;
        midiBuffer = nullptr;
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::reInitFile FILE \"%s\" DOES NOT EXIST OR USER DOES NOT HAVE PERMISSIONS TO READ IT\n", path.c_str());
        return -2;
    }

    this->path = path;

    return 0;
}

char KeyboardRecorder_MidiFile::start(){
    if (running){
        return -1;
    }
    running = true;
    return 0;
}

char KeyboardRecorder_MidiFile::stop(){
    if (running == false){
        return -1;
    }
    running = false;
    return 0;
}

bool KeyboardRecorder_MidiFile::isRunning(){
    return running;
}

char KeyboardRecorder_MidiFile::revind(){
    if (midiBuffer == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::revind CANNOT REWIND WITHOUT INITIALIZATION\n");
        return -1;
    }
    return midiBuffer->fileReader.rewindFile();
}

bool KeyboardRecorder_MidiFile::isReady(){
    if (midiBuffer == nullptr){
        return false;
    }
    return midiBuffer->fileReader.isFileReady();
}

char KeyboardRecorder_MidiFile::play(){
    if (midiBuffer == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::play CANNOT PLAY WITHOUT INITIALIZATION\n");
        return -1;
    }
    if (playing){
        return 0;
    }
    if (midiBuffer->fileReader.eofChunk(0)){
        return -2;
    }

    midiBuffer->fileReader.setObserver(this);

    buffer = midiBuffer;
    playing = true;
    if (playCounter != nullptr){
        (*playCounter) += 1;
    }
    
    return 0;
}

char KeyboardRecorder_MidiFile::pause(){
    if (midiBuffer == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_MidiFile::pause CANNOT PAUSE WITHOUT INITIALIZATION\n");
        return -1;
    }
    if (playing == false){
        return 0;
    }

    buffer = emptyBuffer;
    playing = false;
    if (playCounter != nullptr){
        (*playCounter) -= 1;
    }

    return 0;
}

bool KeyboardRecorder_MidiFile::isPlaying(){
    return playing;
}

void KeyboardRecorder_MidiFile::notifyFileEnd(){
    pause();
}

std::string KeyboardRecorder_MidiFile::getPath(){
    return path;
}

void KeyboardRecorder_MidiFile::setPlayCounter(int* playCounter){
    this->playCounter = playCounter;
}
