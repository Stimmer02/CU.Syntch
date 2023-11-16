#ifndef KEYBOARDINPUT_DEVINPUT
#define KEYBOARDINPUT_DEVINPUT

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
    void getPressedKeys(ushort* keyArr, ushort& count);
    bool getKeyState(ushort key);

    const ushort keyCount;

    private:
    void scannerThreadFunction();

    bool running;
    std::string path;
    std::fstream* inputStream;
    bool* keyStates;
    std::thread* scannerThread;
};
#endif
