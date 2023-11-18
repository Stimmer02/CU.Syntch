#include "SynthUserInterface.h"
#include "Synthesizer.h"
#include "Synthesizer/settings.h"
#include <algorithm>
#include <linux/input-event-codes.h>

void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo){
    std::printf("\33[2J\33[1;1H\33[0mLoop length\n\33[32m   goal: %lius\n   avg:  %.2lfus\n   max:  %lius\n   lat:  %+.2lfus\n\33[0mWork Length\n\33[32m   avg:  %.2lfus\n   max:  %lius\n\33[0mWork Load\n\33[32m   avg:  %3.4lf%%\n   max:  %3.4lf%%\n\33[0mUser Input Latency:\33[32m %.2lfms\n\n\33[0mFormat Info\n   bit depth:   %i b\n   channels:    %i\n   sample rate: %i Hz\n   sample size: %i\n\n\33[31m", pStatistics->loopLength, pStatistics->averageLoopLength, pStatistics->maxLoopLength, pStatistics->averageLoopLatency, pStatistics->averageWorkTime, pStatistics->maxWorkTime, pStatistics->averageLoad, pStatistics->maxLoad, pStatistics->userInputLatency/1000, audioInfo.bitDepth, audioInfo.channels, audioInfo.sampleRate, audioInfo.sampleSize);
}

SynthUserInterface::SynthUserInterface(AudioPipelineSubstitute* audioPipeline, IKeyboardInput* userInput){
    this->audioPipeline = audioPipeline;
    this->userInput = userInput;
     if (userInput->start()){
        delete userInput;
        userInput = nullptr;
    }

    running = false;
    renderMethod = &SynthUserInterface::drawStatistics;
    parseInputMethod = &SynthUserInterface::parseMenuStatistics;
    xPosition = 0;
    yPosition = 0;

    loopDelay = 1000/30;
}

SynthUserInterface::~SynthUserInterface(){
    delete userInput;
}

char SynthUserInterface::start(){

    if (userInput == nullptr){
        return 1;
    }

    audioInfo = *this->audioPipeline->getAudioInfo();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    terminalDiscard.disableInput();
    audioPipeline->start();
    std::printf("ALL RUNNING\n");
    running = true;

    while (this->running){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        parseInput();
        (*this.*renderMethod)();

    }
    audioPipeline->stop();
    std::printf("ALL STOPPED\n");
    terminalDiscard.enableInput();

    return 0;
}

void SynthUserInterface::parseInput(){
    static ushort pressedKeysCount;
    pressedKeysCount = userInput->getPressedKeysCount();

    if (pressedKeysCount == 0){
        return;
    }

    if (userInput->getKeyState(KEY_LEFTCTRL)){
        if (userInput->getKeyState(KEY_Q)){
            running = false;
        } else if (userInput->getKeyState(KEY_1)){
            parseInputMethod = &SynthUserInterface::parseMenuStatistics;
            renderMethod = &SynthUserInterface::drawStatistics;
        } else if (userInput->getKeyState(KEY_2)){
            parseInputMethod = &SynthUserInterface::parseMenuSynthSetting;
            renderMethod = &SynthUserInterface::drawSyntchSettings;
        }
    } else {
        (*this.*parseInputMethod)();
    }
}



void SynthUserInterface::drawSyntchSettings(){
    static const synthesizer::settings* settings = audioPipeline->getSynthSettings(0);
    static char ansi[8][6] = {"\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m", "\33[0m"};
    static int lastYPosition = 0;
    std::strcpy(ansi[lastYPosition], "\33[0m");
    std::strcpy(ansi[yPosition], "\33[7m");
    lastYPosition = yPosition;

    std::printf(
    "\33[2J\33[1;1H"
    "\33[0m\033[1mSTATISTICS┃\033[7mSETTINGS\33[0m\n\n"
    "%s  Pitch: %+2i\n"
    "%s Volume: %2.2f\n"
    "%s Attack: %2.2f\n"
    "%sSustain: %2.2f\n"
    "%s   Fade: %2.2f\n"
    "%sRelease: %2.2f\n\n"
    "%sGenerator type:\n"
    "%s\n",
    ansi[0], settings->pitch, ansi[1], settings->volume, ansi[2], settings->attack.raw, ansi[3], settings->sustain.raw, ansi[4], settings->fade.raw, ansi[5], settings->release.raw, ansi[6], ansi[7]);
}

void SynthUserInterface::drawStatistics(){
    const statistics::pipelineStatistics* pStatistics = audioPipeline->getStatistics();
    std::printf(
    "\33[2J\33[1;1H"
    "\33[0m\033[7m\033[1mSTATISTICS\33[0m\033[1m┃SETTINGS\33[0m\n\n"
    "Loop length\n"
    "\33[32m   goal: %lius\n   avg:  %.2lfus\n   max:  %lius\n   lat:  %+.2lfus\n"
    "\33[0mWork Length\n"
    "\33[32m   avg:  %.2lfus\n   max:  %lius\n"
    "\33[0mWork Load\n"
    "\33[32m   avg:  %3.4lf%%\n   max:  %3.4lf%%\n"
    "\33[0mUser Input Latency:\33[32m %.2lfms\n\n"
    "\33[0mFormat Info\n"
    "   bit depth:   %i b\n   channels:    %i\n   sample rate: %i Hz\n   sample size: %i\n\n\33[31m"
    , pStatistics->loopLength, pStatistics->averageLoopLength, pStatistics->maxLoopLength, pStatistics->averageLoopLatency, pStatistics->averageWorkTime, pStatistics->maxWorkTime, pStatistics->averageLoad, pStatistics->maxLoad, pStatistics->userInputLatency/1000, audioInfo.bitDepth, audioInfo.channels, audioInfo.sampleRate, audioInfo.sampleSize);
}


void SynthUserInterface::parseMenuSynthSetting(){
    static const ushort* pressedKeys = userInput->getPressedKeysArr();
    static const synthesizer::settings* settings = audioPipeline->getSynthSettings(0);
    static const int maxY = 6;
    static const uint inputDelay = loopDelay/1000;

    switch (pressedKeys[0]){
        case KEY_ENTER:

            break;

        case KEY_UP:
            if (yPosition > 0){
                yPosition--;
                while (userInput->getKeyState(KEY_UP)){
                    std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
                    (*this.*renderMethod)();
                }
            }
            break;

        case KEY_DOWN:
            if (yPosition < maxY){
                yPosition++;
                while (userInput->getKeyState(KEY_DOWN)){
                    std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
                    (*this.*renderMethod)();
                }
            }
            break;

        case KEY_LEFT:
            switch (yPosition) {
                case 0:
                    audioPipeline->setSynthSettings(0, synthesizer::PITCH, settings->pitch - 1);
                    break;

                case 1:
                    audioPipeline->setSynthSettings(0, synthesizer::VOLUME, settings->volume - 0.01);
                    break;

                case 2:
                    audioPipeline->setSynthSettings(0, synthesizer::ATTACK, settings->attack.raw - 0.1);
                    break;

                case 3:
                    audioPipeline->setSynthSettings(0, synthesizer::SUSTAIN, settings->sustain.raw - 0.1);
                    break;

                case 4:
                    audioPipeline->setSynthSettings(0, synthesizer::FADE, settings->fade.raw - 0.1);
                    break;

                case 5:
                    audioPipeline->setSynthSettings(0, synthesizer::RELEASE, settings->release.raw - 0.1);
                    break;

                case 6:
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
                    audioPipeline->setSynthSettings(0, synthesizer::ATTACK, settings->attack.raw + 0.1);
                    break;

                case 3:
                    audioPipeline->setSynthSettings(0, synthesizer::SUSTAIN, settings->sustain.raw + 0.1);
                    break;

                case 4:
                    audioPipeline->setSynthSettings(0, synthesizer::FADE, settings->fade.raw + 0.1);
                    break;

                case 5:
                    audioPipeline->setSynthSettings(0, synthesizer::RELEASE, settings->release.raw + 0.1);
                    break;

                case 6:
                    break;

            }
            break;
    }
}


void SynthUserInterface::drawXTimes(uint x){
    for (uint i = 0; i < x; i++){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        (*this.*renderMethod)();
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
