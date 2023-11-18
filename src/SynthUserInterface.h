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
    void drawXTimes(uint x);
    void syntchSettingsChange(const ushort& id, const synthesizer::settings_name& settingsName, const double& value, const uint& keyID);

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

    uint loopDelay;
};

#endif
