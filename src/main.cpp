#include "AudioOutput/audioFormatInfo.h"
#include "AudioPipelineManager.h"
#include "SynthUserInterface.h"
#include "UserInput/InputMap.h"
#include "UserInput/KeyboardInput_DevInput.h"
#include "UserInput/KeyboardRecorder_DevSnd.h"
#include "UserInput/KeyboardRecorder_DevInput.h"
#include "UserInput/MIDI/MidiFileReader.h"
#include "UserInput/TerminalInputDiscard.h"

//AI

void initializeAudioInfoFromFile(audioFormatInfo& audioInfo, const std::string& filename){
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    try {
                        if (key == "BIT_DEPTH") {
                            audioInfo.bitDepth = std::stoi(value);
                        } else if (key == "CHANNELS") {
                            audioInfo.channels = std::stoi(value);
                        } else if (key == "SAMPLE_RATE") {
                            audioInfo.sampleRate = std::stoi(value);
                        } else if (key == "SAMPLE_SIZE") {
                            audioInfo.sampleSize = std::stoi(value);
                        } else if (key == "LITTLE_ENDIAN") {
                            audioInfo.littleEndian = value == "1" or value == "true" or value == "TRUE";
                        }
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "Error: Invalid value for %s. Using default value.\n", key.c_str());
                    }
                }
            }
        }
        audioInfo.byteRate = audioInfo.sampleRate * audioInfo.channels * audioInfo.bitDepth/8;
        audioInfo.blockAlign = audioInfo.channels * audioInfo.bitDepth/8;
    }
}


void updateStatistics(const statistics::pipelineStatistics* pStatistics, const audioFormatInfo& audioInfo);

int main(int argc, char** argv){

    const ushort keyCount = 127;

    audioFormatInfo audioInfo;
    initializeAudioInfoFromFile(audioInfo, "./config/audio.config");
    


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
