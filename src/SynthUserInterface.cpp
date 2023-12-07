#include "SynthUserInterface.h"
#include "AudioPipelineSubstitute.h"
#include "Synthesizer.h"
#include "Synthesizer/IGenerator.h"
#include <linux/input-event-codes.h>


SynthUserInterface::SynthUserInterface(audioFormatInfo audioInfo, AKeyboardRecorder* keyboardInput, IKeyboardInput* userInput, ushort keyCount){
    this->userInput = userInput;
    this->keyboardInput = keyboardInput;
    this->keyCount = keyCount;

    audioPipeline = new AudioPipelineSubstitute(audioInfo, keyCount, keyboardInput);
    if (userInput->start()){
        delete userInput;
        userInput = nullptr;
    }

    running = false;
    renderMethod = &SynthUserInterface::drawSyntchSettings;
    parseInputMethod = &SynthUserInterface::parseMenuSynthSetting;
    xPosition = 0;
    yPosition = 0;

    recordingIndicatorBlink = 0;
    loopDelay = 1000/30;

    fromatSettingsApplied = true;
}

SynthUserInterface::~SynthUserInterface(){
    delete audioPipeline;
}

char SynthUserInterface::start(){
    uint blinkCounter = 0;

    if (userInput == nullptr){
        return 1;
    }

    audioInfo = *audioPipeline->getAudioInfo();
    unappliedAudioInfo = *audioPipeline->getAudioInfo();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    terminalDiscard.disableInput();
    audioPipeline->start();
    std::printf("ALL RUNNING\n");
    running = true;
    toUpdate = true;

    while (this->running){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        parseInput();
        if (audioPipeline->isRecording()){
            blinkCounter++;
            if (blinkCounter > 15){
                blinkCounter = 0;
                recordingIndicatorBlink = !recordingIndicatorBlink;
                toUpdate = true;
            }
        }

        if (toUpdate){
            (*this.*renderMethod)();
        }
    }
    audioPipeline->stop();
    std::printf("ALL STOPPED\n");
    terminalDiscard.enableInput();

    return 0;
}

void SynthUserInterface::parseInput(){
    static ushort pressedKeysCount;
    pressedKeysCount = userInput->getPressedKeysCount();
    toUpdate = true;

    if (pressedKeysCount == 0){
        return;
    }

    if (userInput->getKeyState(KEY_LEFTCTRL)){
        if (userInput->getKeyState(KEY_Q)){
            running = false;
        } else if (userInput->getKeyState(KEY_1)){
            parseInputMethod = &SynthUserInterface::parseMenuStatistics;
            renderMethod = &SynthUserInterface::drawStatistics;
            yPosition = 0;
        } else if (userInput->getKeyState(KEY_2)){
            parseInputMethod = &SynthUserInterface::parseMenuSynthSetting;
            renderMethod = &SynthUserInterface::drawSyntchSettings;
            yPosition = 0;

        } else if (userInput->getKeyState(KEY_3)){
            parseInputMethod = &SynthUserInterface::parseFormatSettings;
            renderMethod = &SynthUserInterface::drawFormatSettings;
            yPosition = 0;

        } else if (userInput->getKeyState(KEY_SPACE)){
            waitUntilKeyReleased(KEY_SPACE);
            if (audioPipeline->isRecording()){
                audioPipeline->stopRecording();
                recordingIndicatorBlink = 0;
            } else {
                audioPipeline->startRecording("test.wav");
            }
        }
    } else {
        (*this.*parseInputMethod)();
    }
}

const std::string recordingMessage[3] = {"\033[1mNOT RECORDING\33[0m", "\033[1m\33[31m  ⏺ RECORDING\33[0m", "\033[1m\33[31m    RECORDING\33[0m"};


void SynthUserInterface::drawSyntchSettings(){
    static const std::string synthNames[3] = {"SINE", "SQARE", "SAWTOOTH"};
    static const synthesizer::settings* settings = audioPipeline->getSynthSettings(0);
    static char ansi[10][6] = {"\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m"};
    static int lastYPosition = 0;
    uint synthType = audioPipeline->getSynthType(0);
    std::strcpy(ansi[lastYPosition], "\33[0m");
    std::strcpy(ansi[yPosition], "\33[7m");
    lastYPosition = yPosition;

    std::printf(
    "\33[2J\33[1;1H"
    "\33[0m\033[1mSTATISTICS┃\033[7mSYNTH SETTINGS\33[0m\033[1m┃AUDIO FORMAT\33[0m\n"
    "%s\n\n"
    "%s      Pitch: %+2i\n"
    "%s     Volume: %2.2f\n"
    "%s Stereo mix: %2.2f\n"
    "%s     Attack: %2.2f\n"
    "%s    Sustain: %2.2f\n"
    "%s       Fade: %2.2f\n"
    "%s    Fade to: %2.2f\n"
    "%s    Release: %2.2f\n\n"
    "%sGenerator type: %s\n"
    "%s\n\n\33[31m",
    recordingMessage[audioPipeline->isRecording()+recordingIndicatorBlink].c_str(), ansi[0], settings->pitch, ansi[1], settings->volume, ansi[2], settings->stereoMix, ansi[3], settings->attack.raw, ansi[4], settings->sustain.raw, ansi[5], settings->fade.raw, ansi[6], settings->rawFadeTo, ansi[7], settings->release.raw, ansi[8], synthNames[synthType].c_str(), ansi[9]);
}

void SynthUserInterface::drawStatistics(){
    toUpdate = true;
    const statistics::pipelineStatistics* pStatistics = audioPipeline->getStatistics();
    std::printf(
    "\33[2J\33[1;1H"
    "\33[0m\033[7m\033[1mSTATISTICS\33[0m\033[1m┃SYNTH SETTINGS┃AUDIO FORMAT\33[0m\n"
    "%s\n\n"
    "Loop length\n"
    "\33[32m   goal: %lius\n   avg:  %.2lfus\n   max:  %lius\n   lat:  %+.2lfus\n"
    "\33[0mWork Length\n"
    "\33[32m   avg:  %.2lfus\n   max:  %lius\n"
    "\33[0mWork Load\n"
    "\33[32m   avg:  %3.4lf%%\n   max:  %3.4lf%%\n"
    "\33[0mUser Input Latency:\33[32m %.2lfms\n\n"
    "\33[0mFormat Info\n"
    "   bit depth:   %i b\n   channels:    %i\n   sample rate: %i Hz\n   sample size: %i\n\n\33[31m",
    recordingMessage[audioPipeline->isRecording()+recordingIndicatorBlink].c_str(), pStatistics->loopLength, pStatistics->averageLoopLength, pStatistics->maxLoopLength, pStatistics->averageLoopLatency, pStatistics->averageWorkTime, pStatistics->maxWorkTime, pStatistics->averageLoad, pStatistics->maxLoad, pStatistics->userInputLatency/1000, audioInfo.bitDepth, audioInfo.channels, audioInfo.sampleRate, audioInfo.sampleSize);
}

void SynthUserInterface::drawFormatSettings(){
    static const std::string applyMessage[2] = {"settings not applied\n\n", ""};
    static char ansi[6][6] = {"\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m"};
    static int lastYPosition = 0;
    std::strcpy(ansi[lastYPosition], "\33[0m");
    std::strcpy(ansi[yPosition], "\33[7m");
    lastYPosition = yPosition;
    std::printf(
    "\33[2J\33[1;1H"
    "\33[0m\033[1mSTATISTICS┃SYNTH SETTINGS┃\033[7mAUDIO FORMAT\33[0m\n"
    "%s\n\n"
    "%s  bit depth: %i b\n"
    "%s   channels: %i\n"
    "%ssample rate: %i Hz\n"
    "%ssample size: %i\n\n"
    "%sAPPLY\n"
    "%sRESET\n\n"
    "\33[0m\33[31m"
    "%s",
    recordingMessage[audioPipeline->isRecording()+recordingIndicatorBlink].c_str(), ansi[0], unappliedAudioInfo.bitDepth, ansi[1], unappliedAudioInfo.channels, ansi[2], unappliedAudioInfo.sampleRate, ansi[3], unappliedAudioInfo.sampleSize, ansi[4], ansi[5], applyMessage[fromatSettingsApplied].c_str());
}

void SynthUserInterface::parseMenuSynthSetting(){
    static const ushort* pressedKeys = userInput->getPressedKeysArr();
    static const synthesizer::settings* settings = audioPipeline->getSynthSettings(0);
    static const int maxY = 8;

    switch (pressedKeys[0]){
        case KEY_ENTER:

            break;

        case KEY_UP:
            if (yPosition > 0){
                yPosition--;
                waitUntilKeyReleased(KEY_UP);
            }
            break;

        case KEY_DOWN:
            if (yPosition < maxY){
                yPosition++;
                waitUntilKeyReleased(KEY_DOWN);
            }
            break;

        case KEY_LEFT:
            switch (yPosition) {
                case 0:
                    audioPipeline->setSynthSettings(0, synthesizer::PITCH, settings->pitch - 1);
                    break;

                case 1:
                    audioPipeline->setSynthSettings(0, synthesizer::VOLUME, settings->volume - 0.01*(settings->volume - 0.01 >= 0));
                    break;

                case 2:
                    audioPipeline->setSynthSettings(0, synthesizer::STEREO, settings->stereoMix - 0.01*(settings->stereoMix - 0.01 >= 0));
                    break;

                case 3:
                    audioPipeline->setSynthSettings(0, synthesizer::ATTACK, settings->attack.raw - 0.1*(settings->attack.raw - 0.1 >= 0));
                    break;

                case 4:
                    audioPipeline->setSynthSettings(0, synthesizer::SUSTAIN, settings->sustain.raw - 0.1*(settings->sustain.raw - 0.1 >= 0));
                    break;

                case 5:
                    audioPipeline->setSynthSettings(0, synthesizer::FADE, settings->fade.raw - 0.1*(settings->fade.raw - 0.1 >= 0));
                    break;

                case 6:
                    audioPipeline->setSynthSettings(0, synthesizer::FADETO, settings->rawFadeTo - 0.1*(settings->rawFadeTo - 0.1 >= 0));
                    break;

                case 7:
                    audioPipeline->setSynthSettings(0, synthesizer::RELEASE, settings->release.raw - 0.1*(settings->release.raw - 0.1 >= 0));
                    break;

                case 8:
                    uint currentType = audioPipeline->getSynthType(0);
                    if (currentType > 0){
                        currentType--;
                        audioPipeline->setSynthSettings(0, synthesizer::generator_type(currentType));
                    }
                    waitUntilKeyReleased(KEY_LEFT);
                    break;
            }
            break;

        case KEY_RIGHT:
            switch (yPosition) {
                case 0:
                    audioPipeline->setSynthSettings(0, synthesizer::PITCH, settings->pitch + 1);
                    break;

                case 1:
                    audioPipeline->setSynthSettings(0, synthesizer::VOLUME, settings->volume + 0.01);
                    break;

                case 2:
                    audioPipeline->setSynthSettings(0, synthesizer::STEREO, settings->stereoMix + 0.01);
                    break;

                case 3:
                    audioPipeline->setSynthSettings(0, synthesizer::ATTACK, settings->attack.raw + 0.1);
                    break;

                case 4:
                    audioPipeline->setSynthSettings(0, synthesizer::SUSTAIN, settings->sustain.raw + 0.1);
                    break;

                case 5:
                    audioPipeline->setSynthSettings(0, synthesizer::FADE, settings->fade.raw + 0.1);
                    break;

                case 6:
                    audioPipeline->setSynthSettings(0, synthesizer::FADETO, settings->rawFadeTo + 0.1);
                    break;

                case 7:
                    audioPipeline->setSynthSettings(0, synthesizer::RELEASE, settings->release.raw + 0.1);
                    break;

                case 8:
                    uint currentType = audioPipeline->getSynthType(0);
                    if (currentType < synthesizer::LAST){
                        currentType++;
                        audioPipeline->setSynthSettings(0, synthesizer::generator_type(currentType));
                    }
                    waitUntilKeyReleased(KEY_RIGHT);
                    break;

            }
            break;
    }
}

template<typename T>
T findIndex(const T arr[], uint conut, T value){
    for (uint i = 0; i < conut; i++){
        if (arr[i] == value){
            return i;
        }
    }
    return 0;
}

void SynthUserInterface::parseFormatSettings(){
    static const ushort* pressedKeys = userInput->getPressedKeysArr();
    static const int maxY = 5;

    static const uint sampleRateOptionsCount = 10;
    static const uint sampleRateOptions[sampleRateOptionsCount] = {
        8000,
        11025,
        16000,
        22050,
        44100,
        48000,
        96000,
        128000,
        176400,
        192000,
    };
    static uint sampleRateCurrent = findIndex<uint>(sampleRateOptions, sampleRateOptionsCount, audioInfo.sampleRate);

    static const uint sampleSizeOptionsCount = 13;
    static const uint sampleSizeOptions[sampleSizeOptionsCount] = {
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536
    };
    static uint sampleSizeCurrent = findIndex<uint>(sampleSizeOptions, sampleSizeOptionsCount, audioInfo.sampleSize);


    static const uint bitDepthOptionsCount = 4;
    static const uchar bitDepthOptions[bitDepthOptionsCount] = {
        8,
        16,
        24,
        32,
    };
    static uint bitDepthCurrent = findIndex<uchar>(bitDepthOptions, bitDepthOptionsCount, audioInfo.bitDepth);

    static const uchar channelsOptionsCount = 2;
    static uchar channelsCurrent = audioInfo.channels;

    switch (pressedKeys[0]){
        case KEY_ENTER:
            switch (yPosition) {
                case 4:
                    if (fromatSettingsApplied == false){
                        audioInfo = unappliedAudioInfo;
                        audioPipeline->stop();
                        delete audioPipeline;
                        keyboardInput->reInit(audioInfo.sampleSize, audioInfo.sampleRate);
                        audioPipeline = new AudioPipelineSubstitute(audioInfo, keyCount, keyboardInput);
                        audioPipeline->start();
                        fromatSettingsApplied = true;
                    }
                    waitUntilKeyReleased(KEY_ENTER);
                    break;

                case 5:
                    if (fromatSettingsApplied == false){
                        unappliedAudioInfo = audioInfo;
                        sampleRateCurrent = findIndex<uint>(sampleRateOptions, sampleRateOptionsCount, audioInfo.sampleRate);
                        sampleSizeCurrent = findIndex<uint>(sampleSizeOptions, sampleSizeOptionsCount, audioInfo.sampleSize);
                        bitDepthCurrent = findIndex<uchar>(bitDepthOptions, bitDepthOptionsCount, audioInfo.bitDepth);
                        channelsCurrent = audioInfo.channels;
                        fromatSettingsApplied = true;
                    }
                    waitUntilKeyReleased(KEY_ENTER);
                    break;
            }
            break;

        case KEY_UP:
            if (yPosition > 0){
                yPosition--;
                waitUntilKeyReleased(KEY_UP);
            }
            break;

        case KEY_DOWN:
            if (yPosition < maxY){
                yPosition++;
                waitUntilKeyReleased(KEY_DOWN);
            }
            break;

        case KEY_LEFT:
            switch (yPosition) {
                case 0:
                    if (bitDepthCurrent > 0){
                        bitDepthCurrent--;
                        unappliedAudioInfo.bitDepth = bitDepthOptions[bitDepthCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_LEFT);
                    break;

                case 1:
                    if (channelsCurrent > 1){
                        channelsCurrent--;
                        unappliedAudioInfo.channels = channelsCurrent;
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_LEFT);
                    break;

                case 2:
                    if (sampleRateCurrent > 0){
                        sampleRateCurrent--;
                        unappliedAudioInfo.sampleRate = sampleRateOptions[sampleRateCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_LEFT);
                    break;

                case 3:
                    if (sampleSizeCurrent > 0){
                        sampleSizeCurrent--;
                        unappliedAudioInfo.sampleSize = sampleSizeOptions[sampleSizeCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_LEFT);
                    break;
            }
            break;

        case KEY_RIGHT:
            switch (yPosition) {
                case 0:
                    if (bitDepthCurrent < bitDepthOptionsCount-1){
                        bitDepthCurrent++;
                        unappliedAudioInfo.bitDepth = bitDepthOptions[bitDepthCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_RIGHT);
                    break;

                case 1:
                    if (channelsCurrent < channelsOptionsCount){
                        channelsCurrent++;
                        unappliedAudioInfo.channels = channelsCurrent;
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_RIGHT);
                    break;

                case 2:
                    if (sampleRateCurrent < sampleRateOptionsCount-1){
                        sampleRateCurrent++;
                        unappliedAudioInfo.sampleRate = sampleRateOptions[sampleRateCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_RIGHT);
                    break;

                case 3:
                    if (sampleSizeCurrent < sampleSizeOptionsCount-1){
                        sampleSizeCurrent++;
                        unappliedAudioInfo.sampleSize = sampleSizeOptions[sampleSizeCurrent];
                        fromatSettingsApplied = false;
                    }
                    waitUntilKeyReleased(KEY_RIGHT);
                    break;

            }
            break;
    }
}

void SynthUserInterface::parseMenuStatistics(){
    // static const ushort* pressedKeys = userInput->getPressedKeysArr();
    // switch (pressedKeys[0]){
    //     case KEY_ENTER:
    //
    //         break;
    // }
}

void SynthUserInterface::waitUntilKeyReleased(ushort key){
    while (userInput->getKeyState(key)){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
         if (toUpdate){
            toUpdate = false;
            (*this.*renderMethod)();
         }
    }
}

void SynthUserInterface::drawXTimes(uint x){
    for (uint i = 0; i < x; i++){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        (*this.*renderMethod)();
    }
}



