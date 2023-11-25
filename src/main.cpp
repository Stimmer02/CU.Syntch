#include "AudioOutput/audioFormatInfo.h"
#include "AudioPipelineSubstitute.h"
#include "SynthUserInterface.h"
#include "UserInput/KeyboardInput_DevInput.h"
#include "UserInput/TerminalInputDiscard.h"

void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo);

int main(int argc, char** argv){

    const ushort keyCount = 32;

    if (argc == 1){
        std::printf("USAGE: %s <event ID>\n", argv[0]);
        return 1;
    }

    AKeyboardRecorder* keyboardInput = new KeyboardRecorder_DevInput(keyCount);
    IKeyboardInput* userInput = new KeyboardInput_DevInput();

    std::string keyboardStreamLocation = "/dev/input/event";
    keyboardStreamLocation += argv[1];


    audioFormatInfo audioInfo;
    audioInfo.bitDepth = 16;
    audioInfo.channels = 1;
    audioInfo.littleEndian = true;
    audioInfo.sampleRate = 48000;
    audioInfo.sampleSize = 512;
    audioInfo.littleEndian = true;
    audioInfo.byteRate = audioInfo.sampleRate * audioInfo.channels * audioInfo.bitDepth/8;
    audioInfo.blockAlign = audioInfo.channels * audioInfo.bitDepth/8;

    InputMap* keyboardMap = new InputMap("./incrementalKeyboardMap.txt");

    if (keyboardInput->init(keyboardStreamLocation, audioInfo.sampleSize, audioInfo.sampleRate, keyboardMap)){
        return 2;
    }

    if (userInput->init(keyboardStreamLocation)){
        return 3;
    }

    SynthUserInterface userInterface(audioInfo, keyboardInput, userInput, keyCount);

    userInterface.start();

    delete keyboardMap;
    delete keyboardInput;
    delete userInput;


    return 0;
}


