#ifndef SYNTHUSERINTERFACE_H
#define SYNTHUSERINTERFACE_H

#include "AudioOutput/audioFormatInfo.h"
#include "UserInput/AKeyboardRecorder.h"
#include "UserInput/IKeyboardInput.h"
#include "AudioPipelineSubstitute.h"
#include "UserInput/TerminalInputDiscard.h"


class SynthUserInterface{
public:
    SynthUserInterface(audioFormatInfo audioInfo, AKeyboardRecorder* keyboardInput, IKeyboardInput* userInput, ushort keyCount);
    ~SynthUserInterface();

    char start();

private:
    void parseInput();

    void drawSyntchSettings();
    void drawStatistics();
    void drawRecordingMenu();
    void drawFormatSettings();
    bool recordingIndicatorBlink;

    void drawXTimes(uint x);
    void waitUntilKeyReleased(ushort key);
    // void syntchSettingsChange(const ushort& id, const synthesizer::settings_name& settingsName, const float& value, const uint& keyID);

    void parseMenuSynthSetting();
    void parseMenuStatistics();
    void parseFormatSettings();
    void parseRecordingMenu();

    typedef void (SynthUserInterface::*methodPtr)();
    methodPtr renderMethod;
    methodPtr parseInputMethod;

    int xPosition;
    int yPosition;


    IKeyboardInput* userInput;
    AKeyboardRecorder* keyboardInput;
    AudioPipelineSubstitute* audioPipeline;
    TerminalInputDiscard terminalDiscard;

    bool running;
    bool toUpdate;
    bool fromatSettingsApplied;
    audioFormatInfo audioInfo;
    audioFormatInfo unappliedAudioInfo;
    ushort keyCount;

    uint loopDelay;
};

#endif
