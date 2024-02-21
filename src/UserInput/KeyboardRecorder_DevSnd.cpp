#include "KeyboardRecorder_DevSnd.h"
#include "MIDI/midiEvent.h"


KeyboardRecorder_DevSnd::KeyboardRecorder_DevSnd(const ushort& keyCount) : keyCount(keyCount){
    buffer = nullptr;
    inputStream = nullptr;
    scannerThread = nullptr;
    running = false;
    path = "";
    sampleRate = 0;
    sampleSize = 0;
}

KeyboardRecorder_DevSnd::~KeyboardRecorder_DevSnd(){
    if (running){
        stop();
    } else {
        if (scannerThread != nullptr){
            delete scannerThread;
        }
    }
    if (inputStream != nullptr){
        inputStream->close();
        delete inputStream;
    }

    if (buffer != nullptr){
        delete buffer;
    }
}

char KeyboardRecorder_DevSnd::init(const std::string path, const uint& sampleSize, const uint& sampleRate){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::init CANNOT INITAILIZE WHILE READIONG THREAD IS RUNNING\n");
        return -1;
    }

    if (inputStream != nullptr){
        delete inputStream;
    }
    inputStream = new std::ifstream(path, std::fstream::in|std::ios::binary);
    this->sampleSize = sampleSize;
    this->sampleRate = sampleRate;

    if (buffer != nullptr){
        delete buffer;
    }
    buffer = new KeyboardDoubleBuffer(sampleSize, keyCount);

    if (!inputStream->is_open()){
        delete inputStream;
        inputStream = nullptr;
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::init FILE \"%s\" DOES NOT EXIST OR USER DOES NOT HAVE PERMISSIONS TO READ IT\n", path.c_str());
        return -2;
    }
    if (inputStream->bad()){
        delete inputStream;
        inputStream = nullptr;
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::init BAD BIT IS SET AFTER OPPENING FILE \"%s\"\n", path.c_str());
        return -3;
    }
    this->path = path;
    return 0;
}

char KeyboardRecorder_DevSnd::reInit(const uint& sampleSize, const uint& sampleRate){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::reInit CANNOT INITAILIZE WHILE READIONG THREAD IS RUNNING\n");
        return -1;
    }

    if (inputStream == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::reInit CANNOT RE-INITAILIZE WITHOUT INITIALIZATION\n");
        return -2;
    }

    this->sampleSize = sampleSize;
    this->sampleRate = sampleRate;

    delete buffer;
    buffer = new KeyboardDoubleBuffer(sampleSize, keyCount);
    return 0;
}

char KeyboardRecorder_DevSnd::start(){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::start READING THREAD WAS ALREADY RUNNING\n");
        return -1;
    }
    if (inputStream == nullptr){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::start CAN NOT START WITHOUT PROPER INITIALIZATION\n");
        return -2;
    }
    if (scannerThread != nullptr){
        delete scannerThread;
    }
    if (inputStream->is_open() == false){
        // inputStream->open(path);
        if (inputStream->bad()){
            delete inputStream;
            inputStream = nullptr;
            std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::start BAD BIT IS SET AFTER OPPENING FILE \"%s\"\n", path.c_str());
            return -3;
        }
    }
    scannerThread = new std::thread(&KeyboardRecorder_DevSnd::scannerThreadFunction, this);
    return 0;
}

char KeyboardRecorder_DevSnd::stop(){
    if (running == false){
        std::fprintf(stderr, "ERR: KeyboardRecorder_DevSnd::stop READING THREAD WAS NOT RUNNIG\n");
        return -1;
    }
    running = false;
    if (scannerThread->joinable()){
        scannerThread->join();
    }
    // inputStream->close();
    delete scannerThread;
    scannerThread = nullptr;
    return 0;
}

bool KeyboardRecorder_DevSnd::isRunning(){
    return running;
}


void KeyboardRecorder_DevSnd::scannerThreadFunction(){
    running = true;
    MIDI::midiEvent event;
    ulong samplePosition;
    ulong sampleLength = 1000000/sampleRate;
    buffer->clearInactiveBuffer();
    buffer->swapActiveBuffer();
    buffer->clearInactiveBuffer();
    while (running){
        interpreter.getEvent(inputStream, event);
        // std::printf("event 0x%02x\n",event.message[0]);
        if (event.type == MIDI::MIDI){
            samplePosition = floor((double(std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count()) - buffer->getActivationTimestamp()) / sampleLength);
            if (samplePosition < sampleSize){
                if (event.message[1] < keyCount){
                    interpreter.executeMidiEvent(event, buffer->getActiveBuffer(), samplePosition);
                } else {
                    // std::printf("UNMAPPED KEY:%d\n", event.message[1]);
                }
            } else {
                if (event.message[1] < keyCount){
                    interpreter.executeMidiEvent(event, buffer->getActiveBuffer(), sampleSize-1);
                    // std::printf("WARNING: KeyboardRecorder_DevSnd::scannerThreadFunction BUFFER WAS NOT SWAPPED FAST ENOUGH\n");
                }
            }
        }
    }
}
