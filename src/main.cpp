#include "AudioOutput/audioFormatInfo.h"
#include "AudioPipelineSubstitute.h"
#include "SynthUserInterface.h"
#include "UserInput/KeyboardInput_DevInput.h"
#include "UserInput/MIDI/MidiFileReader.h"
#include "UserInput/TerminalInputDiscard.h"

#include <sys/poll.h>

void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo);

int main(int argc, char** argv){

    const ushort keyCount = 127;

    audioFormatInfo audioInfo;
    audioInfo.bitDepth = 16;
    audioInfo.channels = 2;
    audioInfo.sampleRate = 48000;
    audioInfo.sampleSize = 512;
    audioInfo.littleEndian = true;
    audioInfo.byteRate = audioInfo.sampleRate * audioInfo.channels * audioInfo.bitDepth/8;
    audioInfo.blockAlign = audioInfo.channels * audioInfo.bitDepth/8;



    struct pollfd fds;
    int ret;
    fds.fd = 0;
    fds.events = POLLIN;
    ret = poll(&fds, 1, 0);
    if (ret == 1){
        std::printf("Reading MIDI file from stdin...\n");

        MIDI::MidiFileReader midiReader(audioInfo.sampleSize, audioInfo.sampleRate);
        AudioPipelineSubstitute audioPipeline(audioInfo, keyCount, nullptr);
        audioPipeline.loadSynthConfig("./config/synth.config", 0);
        audioPipeline.recordUntilStreamEmpty(midiReader);

        return 0;
    }

    if (argc == 3){
        std::printf("Reading MIDI file: %s...\n", argv[1]);

        MIDI::MidiFileReader midiReader(argv[1] ,audioInfo.sampleSize, audioInfo.sampleRate);
        AudioPipelineSubstitute audioPipeline(audioInfo, keyCount, nullptr);
        audioPipeline.loadSynthConfig("./config/synth.config", 0);
        audioPipeline.recordUntilStreamEmpty(midiReader, argv[2]);

        return 0;
    }

    if (argc == 1){
        std::printf("USAGE: %s <event ID>\nor\nUSAGE: %s <MIDI file path> <WAV file out>\nor use stdin and stdout to convert MIDI to WAV\n", argv[0], argv[0]);
        return 1;
    }

    AKeyboardRecorder* keyboardInput = new KeyboardRecorder_DevInput(keyCount);
    IKeyboardInput* userInput = new KeyboardInput_DevInput();

    std::string keyboardStreamLocation = "/dev/input/event";
    keyboardStreamLocation += argv[1];


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


