#ifndef SYNTHUSERINTERFACE_H
#define SYNTHUSERINTERFACE_H

#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/AKeyboardRecorder.h"
#include "UserInput/IKeyboardInput.h"
#include "AudioPipelineSubstitute.h"
#include "UserInput/TerminalInputDiscard.h"

#include <iostream>
#include <map>


class SynthUserInterface{
public:
    SynthUserInterface(audioFormatInfo audioInfo, AKeyboardRecorder* keyboardInput, IKeyboardInput* userInput, ushort keyCount);
    ~SynthUserInterface();

    char start();

private:
    void parseInput();


    IKeyboardInput* userInput;
    AKeyboardRecorder* keyboardInput;
    AudioPipelineSubstitute* audioPipeline;
    TerminalInputDiscard terminalDiscard;

    bool running;
    uint loopDelay;
    ushort keyCount;

    void drawXTimes(uint x);
    void waitUntilKeyReleased(ushort key);


    bool terminalInput;

    std::string inputLine;
    const ushort inputTokenMax = 64;
    const char** inputTokens;
    ushort inputTokenCount;

    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };

    typedef void (SynthUserInterface::*methodPtr)();
    std::map<const char*, methodPtr, cmp_str>* commandMap;

    void initializeCommandMap();

    void commandExit();
    void commandDisable();

};

#endif
