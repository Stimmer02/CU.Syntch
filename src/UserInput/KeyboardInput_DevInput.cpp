#include "KeyboardInput_DevInput.h"

KeyboardInput_DevInput::KeyboardInput_DevInput() : keyCount(512){
    this->keyStates = new bool[keyCount];
    for (uint i = 0; i < keyCount; i++){
        this->keyStates[i] = false;
    }
    running = false;
}

KeyboardInput_DevInput::~KeyboardInput_DevInput(){
    if (running){
        stop();
    } else {
        if (scannerThread != nullptr){
            delete scannerThread;
        }
    }
    if (inputStream != nullptr){
        delete inputStream;
    }
    delete [] keyStates;
}

void KeyboardInput_DevInput::getPressedKeys(ushort* keyArr, ushort& count){
    count = 0;
    for (uint i = 0; i < keyCount; i++){
        if (keyStates[i]){
            keyArr[count] = i;
            count++;
        }
    }
}

char KeyboardInput_DevInput::init(const std::string path){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::initialize CANNOT INITAILIZE WHILE READIONG THREAD IS RUNNING\n");
        return -1;
    }

    if (inputStream != nullptr){
        delete inputStream;
    }
    inputStream = new std::fstream(path, std::fstream::in|std::ios::binary);

    if (!inputStream->is_open()){
        delete inputStream;
        inputStream = nullptr;
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::initialize FILE \"%s\" DOES NOT EXIST OR USER DOES NOT HAVE PERMISSIONS TO READ IT\n", path.c_str());
        return -2;
    }
    if (inputStream->bad()){
        delete inputStream;
        inputStream = nullptr;
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::initialize BAD BIT IS SET AFTER OPPENING FILE \"%s\"\n", path.c_str());
        return -3;
    }

    this->path = path;
    return 0;
}

char KeyboardInput_DevInput::start(){
    if (running){
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::start READING THREAD WAS ALREADY RUNNING\n");
        return -1;
    }
    if (inputStream == nullptr){
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::start CAN NOT START WITHOUT PROPER INITIALIZATION\n");
        return -2;
    }
    if (scannerThread != nullptr){
        delete scannerThread;
    }
    running = true;
    scannerThread = new std::thread(&KeyboardInput_DevInput::scannerThreadFunction, this);

    return 0;
}

char KeyboardInput_DevInput::stop(){
    if (running == false){
        std::fprintf(stderr, "ERR: KeyboardInput_DevInput::stop READING THREAD WAS NOT RUNNIG\n");
        return -1;
    }
    running = false;
    if (scannerThread->joinable()){
        scannerThread->join();
    }
    inputStream->close();
    delete scannerThread;
    scannerThread = nullptr;
    return 0;
}

bool KeyboardInput_DevInput::getKeyState(ushort key){
    return keyStates[key];
}

void KeyboardInput_DevInput::scannerThreadFunction(){
    input_event event;
    while (running){
        inputStream->read(reinterpret_cast<char*>(&event), sizeof(input_event));
        if (event.type == EV_KEY){
            keyStates[event.code] = event.value;
        }
    }
}
