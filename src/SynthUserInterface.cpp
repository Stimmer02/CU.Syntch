#include "SynthUserInterface.h"
#include <cstdio>
#include <linux/input-event-codes.h>
#include <sstream>



SynthUserInterface::SynthUserInterface(std::string terminalHistoryPath, audioFormatInfo audioInfo, AKeyboardRecorder*& keyboardInput, IKeyboardInput*& userInput, ushort keyCount): history(terminalHistoryPath){
    this->userInput = userInput;
    this->keyCount = keyCount;

    inputTokens = new const char*[inputTokenMax];
    initializeCommandMap();

    audioPipeline = new AudioPipelineManager(audioInfo, keyCount);
    audioPipeline->addSynthesizer();
    audioPipeline->loadSynthConfig("./config/synth.config", 0);

    if (audioPipeline->addInput(keyboardInput) < 0){
        std::fprintf(stderr, "COULD NOT START\n");
        return;
    }
    audioPipeline->connectInputToSynth(0, 0);
    audioPipeline->setOutputBuffer(0, pipeline::SYNTH);

    if (userInput->start()){
        delete userInput;
        userInput = nullptr;
        terminalInput = false;
    }
    terminalInput = true;
    specialInputThreadRunning = false;

    running = false;
    loopDelay = 1000/30;
}

SynthUserInterface::~SynthUserInterface(){
    delete[] inputTokens;
    delete audioPipeline;
    delete commandMap;
    delete userInput;
}

char SynthUserInterface::start(){
    if (userInput == nullptr){
        return 1;
    }

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));

    audioPipeline->start();
    uint waitCounter = 0;
    while(audioPipeline->isRuning() == false){
        if (waitCounter > 50){
            std::fprintf(stderr, "ERR: SynthUserInterface::start COULD NOT START AUDIO PIPELINE ON TIME\n");
            return 2;
        }
        waitCounter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    specialInputThread = new std::thread(&SynthUserInterface::specialInputThreadFunction, this);
    std::printf("ALL RUNNING\n");
    running = true;

    audioPipeline->pauseInput();

    while (this->running){
        readInput();
        if (terminalInput){
            parseInput();
        }
    }

    stopSpecialInput();
    audioPipeline->stop();
    userInput->stop();
    std::printf("ALL STOPPED\n");
    return 0;
}

void SynthUserInterface::stopSpecialInput(){
    specialInputThreadRunning = false;
    if (specialInputThread->joinable()){
        specialInputThread->join();//TODO: delete?
    }
}

// void processWithSimulatedInput(const std::string& simulatedInput) {
//     // Save the original cin buffer and cout buffer
//     std::streambuf* originalCin = std::cin.rdbuf();
//     std::streambuf* originalCout = std::cout.rdbuf();
//
//     // Create a stringstream with the simulated input
//     std::istringstream inputBuffer(simulatedInput);
//
//     // Redirect std::cin to use the stringstream buffer
//     std::cin.rdbuf(inputBuffer.rdbuf());
//
//     // You can also redirect std::cout if needed
//     // std::cout.rdbuf(originalCout);
//
//
//     // Restore the original cin and cout buffers
//     std::cin.rdbuf(originalCin);
//     std::cout.rdbuf(originalCout);
// }

void SynthUserInterface::specialInputThreadFunction(){
    specialInputThreadRunning = true;
    while (specialInputThreadRunning){
        std::this_thread::sleep_for(std::chrono::milliseconds(20));//TODO: use mutex here
        uint specialKeys = userInput->getKeyState(KEY_UP) | userInput->getKeyState(KEY_DOWN) << 1;
        if (specialKeys && terminalInput){
            terminalInput = false;
            terminalDiscard.disableInput();
            std::printf("\e[2K\e[G\e[30m\e[107m⮞ ");
            std::string entry;
            if (specialKeys & 0b1){
                entry = history.getPreviousEntry();
                std::printf("%s", entry.c_str());
                waitUntilKeyReleased(KEY_UP);
            } else {
                entry = history.getNextEntry();
                std::printf("%s", entry.c_str());
                waitUntilKeyReleased(KEY_DOWN);
            }
            fflush(stdout);

            bool specialSequence = true;
            while (specialSequence){
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (userInput->getKeyState(KEY_UP)){
                    entry = history.getPreviousEntry();
                    std::printf("\e[2K\e[G\e[30m\e[107m⮞ %s\e[0m", entry.c_str());
                    fflush(stdout);
                    waitUntilKeyReleased(KEY_UP);
                } else if (userInput->getKeyState(KEY_DOWN)){
                    entry = history.getNextEntry();
                    std::printf("\e[2K\e[G\e[30m\e[107m⮞ %s\e[0m", entry.c_str());
                    fflush(stdout);
                    waitUntilKeyReleased(KEY_DOWN);
                } else if (userInput->getKeyState(KEY_ENTER)){
                    std::printf("\n");
                    inputLine = entry;
                    parseInput();
                    waitUntilKeyReleased(KEY_ENTER);
                    specialSequence = false;
                } else if (userInput->getKeyState(KEY_DELETE)){
                    std::printf("\e[2K\e[G");
                    fflush(stdout);
                    waitUntilKeyReleased(KEY_DELETE);
                    specialSequence = false;
                }

            }
            // processWithSimulatedInput("\n\n\n");
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.clear();

            std::printf("\n\e[32m⮞ ");
            fflush(stdout);

            terminalDiscard.enableInput(true);
            terminalInput = true;
            history.resetIndex();
        }
    }
}

void SynthUserInterface::readInput(){
    std::printf("\n\e[97m⮞ ");
    std::getline(std::cin, inputLine);
    std::printf("\e[0m");
}

void SynthUserInterface::parseInput(){
    bool nextElementIsToken = false;
    methodPtr toExecute;

    if (inputLine[0] == '\n'){
        return;
    }

    inputTokenCount = 1;
    inputTokens[0] = inputLine.c_str();

    uint i = 0;
    for (; i < inputLine.length(); i++){
        if (inputLine[i] == ' '){
            inputLine[i] = '\0';
            nextElementIsToken = true;
            break;
        }
    }

    if (inputLine[0] == '\0'){
        return;
    }

    try {
        toExecute = commandMap->at(inputTokens[0]);
    } catch (std::out_of_range const&){
        std::printf("Command \"%s\" does not exist\n", inputTokens[0]);
        inputLine.clear();
        return;
    }
    inputLine[i] = ' ';
    history.addEntry(inputLine);
    inputLine[i] = '\0';

    for (; i < inputLine.length(); i++){
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




    (*this.*toExecute)();
    inputLine.clear();
}

void SynthUserInterface::waitUntilKeyReleased(ushort key){
    while (userInput->getKeyState(key)){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
    }
}

void SynthUserInterface::initializeCommandMap(){
    commandMap = new std::map<const char*, methodPtr, SynthUserInterface::cmp_str>{
        {"exit",     &SynthUserInterface::commandExit},
        {"toggle",   &SynthUserInterface::commandToggle},
        {"help",     &SynthUserInterface::commandHelp},
        {"pStart",   &SynthUserInterface::commandPipelineStart},
        {"pStop",    &SynthUserInterface::commandPipelineStop},
        {"midiRec",  &SynthUserInterface::commandMidiRecord},
        {"clear",  &SynthUserInterface::commandClear},

        {"synthSave",&SynthUserInterface::commandSynthSave},
        {"synthAdd",&SynthUserInterface::commandSynthAdd},
        {"synthRemove",&SynthUserInterface::commandSynthRemove},
        {"synthCount",&SynthUserInterface::commandSynthCount},
        {"synthConnect",&SynthUserInterface::commandSynthConnect},
        {"synthDisconnect",&SynthUserInterface::commandSynthDisconnect},

        {"inputAdd",&SynthUserInterface::commandInputAdd},
        {"inputRemove",&SynthUserInterface::commandInputRemove},
        {"inputCount",&SynthUserInterface::commandInputCount},

    };
}


void SynthUserInterface::commandExit(){
    running = false;
    std::printf("System shuting down...\n");
}

void SynthUserInterface::commandToggle(){
    terminalDiscard.disableInput();
    audioPipeline->reausumeInput();
    terminalInput = false;
    std::printf("Terminal input disabled, to neable it press \"Ctrl+Q\"\n");
    while (terminalInput == false){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        if (userInput->getKeyState(KEY_LEFTCTRL) && userInput->getKeyState(KEY_Q)){
            waitUntilKeyReleased(KEY_Q);
            terminalInput = true;
        }
    }
    audioPipeline->pauseInput();
    terminalDiscard.enableInput();
    std::printf("Terminal input enabled\n");
}

void SynthUserInterface::commandHelp(){//TODO
    static const std::string help =
    "HELP PROMPT:\n"
    "Usage: command <arguments>\n"
    "\n"
    "command list:\n"
    "\n"
    "help    - shows this message\n"
    "toggle  - toggles input between synthesizer and console, after switching to synthesizer press Ctrl+Q to switch back\n"
    "exit    - exits the program (if program does not turn off: press any key on every connected device to end keyboard input reading threads)\n"
    "pStart  - starts audio pipeline\n"
    "pStop   - stops audio pipeline\n"
    "midiRec <midi file path> <output name> <synth ID> - reads MIDI file and records it to specified .WAV file using specific synthesizer\n"
    "\n"
    "synthSave <load/save> <save file path> <synth ID> - loads or saves synthesizer configuration\n"
    "synthAdd - adds new synthesizer and returns its ID\n"
    "synthRemove <synth ID> - removes synthesizer by its ID\n"
    "synthCount - returns the total count of the synthesizers\n"
    "synthConnect <synth ID> <inputID> - connects specified synthesizer with specified input so the synth will receive keyboard state from that input\n"
    "synthDisconnect <synth ID> - removes the connection so the synthesizer wont't be used until new data stream is connected\n"
    "\n"
    "inputAdd <type> <stream path> <key count> <optional: key map file path> - adds new input and returns its ID, where the type is 'keyboard' or 'midi', stream path describes stream location, key count tells how many notes it should use, key map file path (if type is keybard) describes map file location that will provide keyboard layout interpretation\n"
    "inputRemove <input ID> - removes input by its I\n"
    "inputCount - returns the total count of the inputs\n"
    "\n"
    "\n";
    std::printf("%s\n", help.c_str());
}

void SynthUserInterface::commandPipelineStart(){
    if(audioPipeline->start()){
        std::printf("Couldn't start pipeline!\n");
        return;
    }
    if (terminalInput == false){
        audioPipeline->pauseInput();
    }
    std::printf("Pipeline running\n");
}

void SynthUserInterface::commandPipelineStop(){
    audioPipeline->stop();
    std::printf("Pipeline stopped\n");
}

void SynthUserInterface::commandMidiRecord(){
    if (inputTokenCount < 4){
        std::printf("Usage: midiRec <midi file path> <output name> <synth ID>\n");
        return;
    }
    if (audioPipeline->isRuning()){
        std::printf("To continue audio pipeline have to be idle (run pStop)\n");
        return;
    }

    short synthID = std::stoi(inputTokens[3]);
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("Given synthesizer ID (%i) is not valid\n", synthID);
        return;
    }

    std::printf("Reading MIDI file: %s\n", inputTokens[1]);
    MIDI::MidiFileReader midiReader(inputTokens[1] , audioPipeline->getAudioInfo()->sampleSize, audioPipeline->getAudioInfo()->sampleRate);

    auto start = std::chrono::high_resolution_clock::now();
    if (audioPipeline->recordUntilStreamEmpty(midiReader, synthID, inputTokens[2])){
        std::printf("Something went wrong!\n");
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::printf("File successfully saved as: %s\n", inputTokens[2]);
    std::printf("Time elapsed: %f s\n", std::chrono::duration<double>(end-start).count());
}

void SynthUserInterface::commandClear(){
    system("clear");
}

void SynthUserInterface::commandSynthSave(){
    if (inputTokenCount < 4){
        std::printf("Usage: synthSave <load/save> <save file path> <synth ID>\n");
        return;
    }
    short synthID = std::stoi(inputTokens[3]);
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("Given synthesizer ID (%i) is not valid\n", synthID);
        return;
    }

    if (std::strcmp("load", inputTokens[1]) == 0){
        if (audioPipeline->loadSynthConfig(inputTokens[2], synthID)){
            std::printf("Something went wrong!\n");
            return;
        }
        std::printf("Configuration loaded\n");
    } else if (std::strcmp("save", inputTokens[1]) == 0){
        if (audioPipeline->saveSynthConfig(inputTokens[2], synthID)){
            std::printf("Something went wrong!\n");
            return;
        }
        std::printf("Configuration saved\n");
    } else {
        std::printf("Unknown option: %s\n", inputTokens[1]);
    }
}

void SynthUserInterface::commandSynthAdd(){
    short synthID = audioPipeline->addSynthesizer();
    std::printf("Synth created with ID: %d\n", synthID);
}

void SynthUserInterface::commandSynthRemove(){
    if (inputTokenCount < 2){
        std::printf("Usage: synthRemove <synth ID>\n");
        return;
    }
    short synthID = std::stoi(inputTokens[1]);
    if (audioPipeline->removeSynthesizer(synthID)){
        std::printf("Something went wrong!\n");
        return;
    }
    std::printf("Synth (%d) removed\n", synthID);
}

void SynthUserInterface::commandSynthCount(){
    short count = audioPipeline->getSynthesizerCount();
    std::printf("Synth count: %d\n", count);
}

void SynthUserInterface::commandInputAdd(){
    if (inputTokenCount < 4){
        std::printf("Usage: inputAdd <type> <stream path> <key count>\n");
        return;
    }

    AKeyboardRecorder* newInput;
    if (std::strcmp(inputTokens[2], "keyboard") == 0){
        if (inputTokenCount < 5){
            std::printf("Usage: inputAdd keyboard <stream path> <key count> <key map file path>\n");
            return;
        }
        std::printf("Interpreting as keyboard input\n");
        InputMap map(inputTokens[5]);
        ushort mapKeyCount = map.getKeyCount();
        ushort userKeyCount = std::stoi(inputTokens[4]);
        ushort inputKeyCount = userKeyCount;

        if (keyCount < inputKeyCount){
            std::printf("System configuration does not support specified key count: %d\n system will use: %d\n", userKeyCount, keyCount);
            inputKeyCount = keyCount;
        }

        if (mapKeyCount < inputKeyCount){
            std::printf("Provided key map does not support specified key count: %d\n system will use: %d\n", userKeyCount, mapKeyCount);
            inputKeyCount = mapKeyCount;
        }
        newInput = new KeyboardRecorder_DevInput(inputKeyCount, &map);

    } else if (std::strcmp(inputTokens[2], "midi") == 0){
        std::printf("Interpreting as midi input\n");
        ushort userKeyCount = std::stoi(inputTokens[4]);
        ushort inputKeyCount = userKeyCount;

        if (keyCount < inputKeyCount){
            std::printf("System configuration does not support specified key count: %d\n system will use: %d\n", userKeyCount, keyCount);
            inputKeyCount = keyCount;
        }
        newInput = new KeyboardRecorder_DevSnd(inputKeyCount);

    } else {
        std::printf("Not known device type: %s\n", inputTokens[2]);
        return;
    }

    newInput->init(inputTokens[3], audioPipeline->getAudioInfo()->sampleSize, audioPipeline->getAudioInfo()->sampleRate);

    short inputID = audioPipeline->addInput(newInput);
    std::printf("Input created with ID: %d\n", inputID);
}

void SynthUserInterface::commandInputRemove(){
    if (inputTokenCount < 2){
        std::printf("Usage: inputRemove <synth ID>\n");
        return;
    }
    short inputID = std::stoi(inputTokens[1]);
    if (audioPipeline->removeInput(inputID)){
        std::printf("Something went wrong!\n");
        return;
    }
    std::printf("Input (%d) removed\n", inputID);
}

void SynthUserInterface::commandInputCount(){
    short count = audioPipeline->getInputCount();
    std::printf("Input count: %d\n", count);
}

void SynthUserInterface::commandSynthConnect(){
    if (inputTokenCount < 3){
        std::printf("Usage: synthConnect <synth ID> <inputID>\n");
        return;
    }
    short synthID = std::stoi(inputTokens[1]);
    short inputID = std::stoi(inputTokens[2]);
    if (audioPipeline->connectInputToSynth(inputID, synthID)){
        std::printf("Something went wrong!\n");
        return;
    }
    std::printf("Connected synth (%d) <- input (%d)\n", synthID, inputID);
}

void SynthUserInterface::commandSynthDisconnect(){
    if (inputTokenCount < 2){
        std::printf("Usage: synthConnect <synth ID>\n");
        return;
    }
    short synthID = std::stoi(inputTokens[1]);
    if (audioPipeline->disconnectSynth(synthID)){
        std::printf("Something went wrong!\n");
        return;
    }
    std::printf("Disconnected synth (%d)\n", synthID);
}

