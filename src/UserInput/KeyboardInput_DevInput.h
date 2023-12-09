#ifndef KEYBOARDINPUT_DEVINPUT_H
#define KEYBOARDINPUT_DEVINPUT_H

#include <sys/types.h>
#include <thread>
#include <string>
#include <fstream>
#include <linux/input.h>

#include "IKeyboardInput.h"

class KeyboardInput_DevInput :public IKeyboardInput{
    public:
    KeyboardInput_DevInput();
    ~KeyboardInput_DevInput();
    char init(const std::string path);
    char start();
    char stop();
    bool getKeyState(ushort key);
    const ushort* getPressedKeysArr();
    ushort getPressedKeysCount();
    ushort getKeyCount();

    const ushort keyCount;

    private:
    void scannerThreadFunction();

    ushort* pressedKeysArr;
    ushort pressedKeysCount;

    bool running;
    std::string path;
    std::fstream* inputStream;
    ushort* keyStates;
    std::thread* scannerThread;
};
#endif
