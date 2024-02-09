#ifndef SYNTHUSERINTERFACE_H
#define SYNTHUSERINTERFACE_H

#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/AKeyboardRecorder.h"
#include "UserInput/IKeyboardInput.h"
#include "AudioPipelineManager.h"
#include "UserInput/TerminalInputDiscard.h"
#include "UserInput/TerminalHistory.h"
#include "UserInput/InputMap.h"
#include "UserInput/KeyboardRecorder_DevInput.h"
#include "UserInput/KeyboardRecorder_DevSnd.h"
// #include "ScriptExecutor.h" //TODO

#include <linux/input-event-codes.h>
#include <iostream>
#include <map>


class SynthUserInterface{
public:
    SynthUserInterface(std::string terminalHistoryPath, audioFormatInfo audioInfo, AKeyboardRecorder*& keyboardInput, IKeyboardInput*& userInput, ushort keyCount);
    ~SynthUserInterface();

    char start();

private:
    void parseInput();
    void readInput();
    void waitUntilKeyReleased(ushort key);

    IKeyboardInput* userInput;
    AudioPipelineManager* audioPipeline;
    TerminalInputDiscard terminalDiscard;
    TerminalHistory history;

    bool running;
    uint loopDelay;
    ushort keyCount;


    bool terminalInput;

    std::string inputLine;
    const ushort inputTokenMax = 64;
    const char** inputTokens;
    ushort inputTokenCount;

    void specialInput();

    struct cmp_str{
        bool operator()(const char* a, const char* b) const{
            return std::strcmp(a, b) < 0;
        }
    };

    typedef void (SynthUserInterface::*methodPtr)();
    std::map<const char*, methodPtr, cmp_str>* commandMap;

    void initializeCommandMap();

    void commandExit();
    void commandToggle();
    void commandHelp();
    void commandPipelineStart();
    void commandPipelineStop();
    void commandMidiRecord();
    void commandExecuteScript();//TODO IMPORTANT
    void commandSetOutputBuffer();//TODO
    void commandClear();

    void commandSynthSave();
    void commandSynthAdd();
    void commandSynthRemove();
    void commandSynthCount();
    void commandSynthModify();//TODO
    void commandSynthSettings();//TODO
    void commandSynthList();//TODO

    void commandInputAdd();
    void commandInputRemove();
    void commandInputCount();
    void commandInputList();//TODO

    void commandSynthConnect();
    void commandSynthDisconnect();
    void commandReinitializeID();//TODO


};

#endif
