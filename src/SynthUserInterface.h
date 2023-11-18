#ifndef _SYNTHUSERINTERFACE_H
#define _SYNTHUSERINTERFACE_H

#include "UserInput/IKeyboardInput.h"
#include "AudioPipelineSubstitute.h"
#include "UserInput/TerminalInputDiscard.h"


class SynthUserInterface{
public:
    SynthUserInterface(AudioPipelineSubstitute* audioPipeline, IKeyboardInput* userInput);
    ~SynthUserInterface();

    char start();

private:
    void parseInput();

    void drawSyntchSettings();
    void drawStatistics();

    void parseMenuSynthSetting();
    void parseMenuStatistics();

    typedef void (SynthUserInterface::*methodPtr)();
    methodPtr renderMethod;
    methodPtr parseInputMethod;

    int xPosition;
    int yPosition;


    IKeyboardInput* userInput;
    AudioPipelineSubstitute* audioPipeline;
    TerminalInputDiscard terminalDiscard;

    bool running;
    audioFormatInfo audioInfo;
};

#endif
