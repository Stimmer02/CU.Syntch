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


int main(int argc, char** argv){

    const ushort keyCount = 127;

    audioFormatInfo audioInfo;
    initializeAudioInfoFromFile(audioInfo, "./config/audio.config");
    


    if (argc > 2 && (std::strcmp("help", argv[1]) == 0 || std::strcmp("--help", argv[1]) == 0)){
        std::printf(
        "USAGE: \n"
        "%s (no arguments)     - launches default script in user mode\n"
        "%s <script path>      - launches script and closes at the end\n"
        "%s user <script path> - launches script and stays open\n",
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
        if (std::strcmp("user", argv[1])){
            std::fprintf(stderr, "Invalid argument: %s\n", argv[1]);
            return 1;
        }
        pipeline::SynthUserInterface userInterface("./config/histSave.txt", audioInfo, keyCount);
        std::printf("Executing script: %s\n", argv[2]);
        if (userInterface.scriptReader.executeScript(argv[2], true)){
            std::fprintf(stderr, "%s", userInterface.scriptReader.getLastError().c_str());
            return 1;
        }
        userInterface.start();
    }

    return 0;
}
