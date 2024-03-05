#include "AudioOutput/audioFormatInfo.h"
#include "AudioPipelineManager.h"
#include "SynthUserInterface.h"
#include "UserInput/InputMap.h"
#include "UserInput/KeyboardInput_DevInput.h"
#include "UserInput/KeyboardRecorder_DevSnd.h"
#include "UserInput/KeyboardRecorder_DevInput.h"
#include "UserInput/MIDI/MidiFileReader.h"
#include "UserInput/TerminalInputDiscard.h"


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


    if (argc == 2 && (std::strcmp("help", argv[1]) || std::strcmp("--help", argv[1]))){
        std::printf(
        "USAGE: \n"
        "%s (no arguments)\n"
        "%s <script path>\n"
        "%s <MIDI file path> <WAV file out>\n",
        argv[0], argv[0], argv[0]);
    } else if (argc == 1){
        pipeline::SynthUserInterface userInterface("./config/histSave.txt", audioInfo, keyCount);
        if (userInterface.scriptReader.executeScript("./config/scripts/default.txt", true)){
            std::fprintf(stderr, "%s", userInterface.scriptReader.getLastError().c_str());
            return 1;
        }
        userInterface.start();
    } else if (argc == 2){
        pipeline::SynthUserInterface userInterface("./config/histSave.txt", audioInfo, keyCount);
        std::printf("Executing script: %s\n", argv[1]);
        if (userInterface.scriptReader.executeScript(argv[1], true)){
            std::fprintf(stderr, "%s", userInterface.scriptReader.getLastError().c_str());
            return 1;
        }
    } else {
        std::printf("Reading MIDI file: %s\n", argv[1]);

        MIDI::MidiFileReader midiReader(argv[1] ,audioInfo.sampleSize, audioInfo.sampleRate);
        pipeline::AudioPipelineManager audioPipeline(audioInfo, keyCount);
        short synthID = audioPipeline.addSynthesizer();
        audioPipeline.loadSynthConfig("./config/synth.config", synthID);

        auto start = std::chrono::high_resolution_clock::now();
        if (audioPipeline.recordUntilStreamEmpty(midiReader, synthID, argv[2])){
            return 2;
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::printf("File successfully saved as: %s\n", argv[2]);
        std::printf("Time elapsed: %f s\n", std::chrono::duration<double>(end-start).count());
    }

    return 0;
}
