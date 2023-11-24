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
    void drawRecordingMenu();
    bool recordingIndicatorBlink;

    void drawXTimes(uint x);
    void waitUntilKeyReleased(ushort key);
    void syntchSettingsChange(const ushort& id, const synthesizer::settings_name& settingsName, const double& value, const uint& keyID);

    void parseMenuSynthSetting();
    void parseMenuStatistics();
    void parseRecordingMenu();

    typedef void (SynthUserInterface::*methodPtr)();
    methodPtr renderMethod;
    methodPtr parseInputMethod;

    int xPosition;
    int yPosition;


    IKeyboardInput* userInput;
    AudioPipelineSubstitute* audioPipeline;
    TerminalInputDiscard terminalDiscard;

    bool running;
    bool toUpdate;
    audioFormatInfo audioInfo;

    uint loopDelay;
};

#endif
