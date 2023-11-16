#include "AudioOutput/audioFormatInfo.h"
#include "AudioPipelineSubstitute.h"
#include "UserInput/KeyboardInput_DevInput.h"
#include "UserInput/TerminalInputDiscard.h"

void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo);

int main(int argc, char** argv){

    const ushort keyCount = 128;

    if (argc == 1){
        std::printf("USAGE: %s <event ID>\n", argv[0]);
        return 1;
    }

    TerminalInputDiscard terminal;
    terminal.disableInput();

    AKeyboardRecorder* keyboardInput = new KeyboardRecorder_DevInput(keyCount);
    IKeyboardInput* userInput = new KeyboardInput_DevInput();

    std::string keyboardStreamLocation = "/dev/input/event";
    keyboardStreamLocation += argv[1];

    audioFormatInfo audioInfo;
    audioInfo.bitDepth = 16;
    audioInfo.channels = 1;
    audioInfo.littleEndian = true;
    audioInfo.sampleRate = 96000;
    audioInfo.sampleSize = 512;

    InputMap* keyboardMap = new InputMap("./incrementalKeyboardMap.txt");

    if (keyboardInput->init(keyboardStreamLocation, audioInfo.sampleSize, audioInfo.sampleRate, keyboardMap)){
        return 2;
    }
    if (userInput->init(keyboardStreamLocation)){
        return 3;
    }
    if (userInput->start()){
        return 4;
    }
    AudioPipelineSubstitute pipeline(audioInfo, keyCount, keyboardInput);

    system("clear");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    uint counter = 0;

    pipeline.start();
    std::printf("ALL RUNNING\n");
    while (userInput->getKeyState(KEY_ENTER) == false){
        std::this_thread::sleep_for(std::chrono::milliseconds(100/3));
        counter++;
        if (counter == 15){
            counter = 0;
            updateStatistics(pipeline.getStatistics(), audioInfo);
        }
    }
    pipeline.stop();
    std::printf("ALL STOPPED\n");

    delete userInput;
    delete keyboardInput;
    delete keyboardMap;

    terminal.enableInput();
    return 0;
}

void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo){
    std::printf("\33[2J\33[1;1H\x1b[0mLoop length\n\x1b[32m   goal: %lius\n   avg:  %.2lfus\n   max:  %lius\n   lat:  %+.2lfus\n\x1b[0mWork Length\n\x1b[32m   avg:  %.2lfus\n   max:  %lius\n\x1b[0mWork Load\n\x1b[32m   avg:  %3.4lf%%\n   max:  %3.4lf%%\n\x1b[0mUser Input Latency:\x1b[32m %.2lfms\n\n\x1b[0mFormat Info\n   bit depth:   %i b\n   channels:    %i\n   sample rate: %i Hz\n   sample size: %i\n\n\x1b[31m", pStatistics->loopLength, pStatistics->averageLoopLength, pStatistics->maxLoopLength, pStatistics->averageLoopLatency, pStatistics->averageWorkTime, pStatistics->maxWorkTime, pStatistics->averageLoad, pStatistics->maxLoad, pStatistics->userInputLatency/1000, audioInfo.bitDepth, audioInfo.channels, audioInfo.sampleRate, audioInfo.sampleSize);
}
