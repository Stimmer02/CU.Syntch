#include "SynthUserInterface.h"
#include "UserInput/InputMap.h"


SynthUserInterface::SynthUserInterface(audioFormatInfo audioInfo, AKeyboardRecorder* keyboardInput, IKeyboardInput* userInput, ushort keyCount){
    this->userInput = userInput;
    this->keyboardInput = keyboardInput;
    this->keyCount = keyCount;

    inputTokens = new const char*[inputTokenMax];
    initializeCommandMap();

    audioPipeline = new AudioPipelineSubstitute(audioInfo, keyCount, keyboardInput);
    audioPipeline->loadSynthConfig("./config/synth.config", 0);

    if (userInput->start()){
        delete userInput;
        userInput = nullptr;
    }

    running = false;

    loopDelay = 1000/30;
}

SynthUserInterface::~SynthUserInterface(){
    delete[] inputTokens;
    delete audioPipeline;
    delete commandMap;
}

char SynthUserInterface::start(){
    if (userInput == nullptr){
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    audioPipeline->start();
    std::printf("ALL RUNNING\n");
    running = true;

    while (this->running){
        std::printf("\n> ");
        parseInput();
    }

    audioPipeline->stop();
    userInput->stop();
    std::printf("ALL STOPPED\n");

    return 0;
}

void SynthUserInterface::parseInput(){
    bool nextElementIsToken = false;

    std::getline(std::cin, inputLine);

    inputTokenCount = 1;
    inputTokens[0] = inputLine.c_str();
    for (uint i = 0; i < inputLine.length(); i++){
        if (inputLine[i] == ' '){
            inputLine[i] = '\0';
            nextElementIsToken = true;
        } else if (nextElementIsToken){
            nextElementIsToken = false;
            inputTokens[inputTokenCount] = &inputLine[i];
            inputTokenCount++;
            if (inputTokenCount == inputTokenMax){
                break;
            }
        }
    }

    methodPtr toExecute;
    try {
        toExecute = commandMap->at(inputTokens[0]);
    } catch (std::out_of_range const&){
        std::printf("Command \"%s\" does not exist\n", inputTokens[0]);
        return;
    }

    (*this.*toExecute)();
}

void SynthUserInterface::waitUntilKeyReleased(ushort key){
    while (userInput->getKeyState(key)){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
    }
}

void SynthUserInterface::initializeCommandMap(){
    commandMap = new std::map<const char*, methodPtr, SynthUserInterface::cmp_str>{
        {"exit",    &SynthUserInterface::commandExit},
        {"disable", &SynthUserInterface::commandDisable},
    };
}


void SynthUserInterface::commandExit(){
    running = false;
    std::printf("System shuting down...\n");
}

void SynthUserInterface::commandDisable(){
    terminalDiscard.disableInput();
    terminalInput = false;
    std::printf("Terminal input disabled, to neable it press \"Ctrl+Q\"\n");
    while (terminalInput == false){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        if (userInput->getKeyState(KEY_LEFTCTRL) && userInput->getKeyState(KEY_Q)){
            waitUntilKeyReleased(KEY_Q);
            terminalInput = true;
        }
    }
    terminalDiscard.enableInput();
    std::printf("Terminal input enabled\n");
}
