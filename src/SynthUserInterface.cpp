#include "SynthUserInterface.h"
#include "Pipeline/IDManager.h"
#include "StringToEnum.h"
#include "Synthesizer.h"
#include "Synthesizer/AGenerator.h"
#include "UserInput/ScriptReader.h"
#include <cstdio>



SynthUserInterface::SynthUserInterface(std::string terminalHistoryPath, audioFormatInfo audioInfo, AKeyboardRecorder*& keyboardInput, IKeyboardInput*& userInput, ushort keyCount): scriptReader(this), history(terminalHistoryPath){
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

    running = false;
    error = false;
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
    while (audioPipeline->isRuning() == false){
        if (waitCounter > 64){
            std::fprintf(stderr, "ERR: SynthUserInterface::start COULD NOT START AUDIO PIPELINE ON TIME\n");
            return 2;
        }
        waitCounter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
    }
    running = true;


    std::printf("ALL RUNNING\n");

    audioPipeline->pauseInput();

    while (this->running){
        readInput();
        parseInput();
    }

    audioPipeline->stop();
    userInput->stop();
    std::printf("ALL STOPPED\n");
    return 0;
}

void SynthUserInterface::browseHistory(){
    terminalDiscard.disableInput();
    std::printf("\e[A\e[2K\e[G\e[30m\e[107m⮞ ");
    std::string entry = history.getPreviousEntry();
    std::printf("%s", entry.c_str());
    fflush(stdout);
    waitUntilKeyReleased(KEY_ENTER);

    bool specialSequence = true;
    while (specialSequence){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        if (userInput->getKeyState(KEY_UP)){
            entry = history.getPreviousEntry();
            std::printf("e[0m\e[2K\e[G\e[30m\e[107m⮞ %s\e[0m", entry.c_str());
            fflush(stdout);
            waitUntilKeyReleased(KEY_UP);
        } else if (userInput->getKeyState(KEY_DOWN)){
            entry = history.getNextEntry();
            std::printf("e[0m\e[2K\e[G\e[30m\e[107m⮞ %s\e[0m", entry.c_str());
            fflush(stdout);
            waitUntilKeyReleased(KEY_DOWN);
        } else if (userInput->getKeyState(KEY_ENTER)){
            std::printf("\n");
            inputLine = entry;
            parseInput();
            std::printf("\n\e[30m\e[107m⮞ %s", entry.c_str());
            fflush(stdout);
            // specialSequence = false;
            waitUntilKeyReleased(KEY_ENTER);
        } else if (userInput->getKeyState(KEY_DELETE)){
            std::printf("\e[2K\e[G");
            fflush(stdout);
            specialSequence = false;
            waitUntilKeyReleased(KEY_DELETE);
        }
    }
    terminalDiscard.enableInput(true);

    history.resetIndex();

}

void SynthUserInterface::readInput(){
    std::printf("\n\e[97m⮞ ");
    fflush(stdout);
    std::getline(std::cin, inputLine);
    std::printf("\e[0m");
}

void SynthUserInterface::parseInput(){
    bool nextElementIsToken = false;
    methodPtr toExecute;

    if (inputLine[0] == '\n'){
        return;
    }

    if (inputLine[0] == ' '){
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
    std::printf("\n");

    try {
        toExecute = commandMap->at(inputTokens[0]);
    } catch (std::out_of_range const&){
        std::printf("Command \"%s\" does not exist\n", inputTokens[0]);
        inputLine.clear();
        error = true;
        return;
    }

    if (inputLine[0] != '\e'){
        inputLine[i] = ' ';
        history.addEntry(inputLine);
        inputLine[i] = '\0';
    }

    i++;

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
    std::printf("\n");

    (*this.*toExecute)();
    inputLine.clear();
}

bool SynthUserInterface::getErrorFlag(){
    return error;
}

void SynthUserInterface::clearErrorFlag(){
    error = false;
}

void SynthUserInterface::executeCommand(std::string& command){
    inputLine = command;
    parseInput();
}



void SynthUserInterface::waitUntilKeyReleased(ushort key){
    while (userInput->getKeyState(key)){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
    }
}

template <typename INTEGER, typename>
char SynthUserInterface::numberFromToken(short tokenIndex, INTEGER& out){
    try {
        out = std::stoi(inputTokens[tokenIndex]);
    } catch (const std::invalid_argument&){
        std::printf("Could not parse %s, as inteager\n", inputTokens[tokenIndex]);
        return -1;
    } catch (const std::out_of_range&){
        std::printf("Could not parse %s, as inteager\n", inputTokens[tokenIndex]);
        return -2;
    }
    return 0;
}

char SynthUserInterface::numberFromToken(short tokenIndex, float& out){
    try {
        out = std::stof(inputTokens[tokenIndex]);
    } catch (const std::invalid_argument&){
        std::printf("Could not parse %s, as floating point number\n", inputTokens[tokenIndex]);
        return -1;
    } catch (const std::out_of_range&){
        std::printf("Could not parse %s, as floating point number\n", inputTokens[tokenIndex]);
        return -2;
    }
    return 0;
}

void SynthUserInterface::stopPipeline(){
    if (audioPipeline->isRuning()){
        std::printf("Can not execute this action if pipeline is running\n");
        audioPipeline->stop();
        std::printf("Pipeline stopped\n");
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
        {"setOut",&SynthUserInterface::commandSetOutputBuffer},
        {"idReinit",&SynthUserInterface::commandReinitializeID},
        {"execute",&SynthUserInterface::commandExecuteScript},

        {"\e[A", &SynthUserInterface::browseHistory},

        {"synthSave",&SynthUserInterface::commandSynthSave},
        {"synthAdd",&SynthUserInterface::commandSynthAdd},
        {"synthRemove",&SynthUserInterface::commandSynthRemove},
        {"synthCount",&SynthUserInterface::commandSynthCount},
        {"synthConnect",&SynthUserInterface::commandSynthConnect},
        {"synthDisconnect",&SynthUserInterface::commandSynthDisconnect},
        {"synthGet",&SynthUserInterface::commandSynthSettings},
        {"synthSet",&SynthUserInterface::commandSynthModify},

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
    if (audioPipeline->isRuning() == false){
        std::printf("Pipeline is not running\n");
        error = true;
        return;
    }
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
    "help    - shows this message\n\n"
    "toggle  - toggles input between synthesizer and console, after switching to synthesizer press Ctrl+Q to switch back\n\n"
    "exit    - exits the program (if program does not turn off: press any key on every connected device to end keyboard input reading threads)\n\n"
    "pStart  - starts audio pipeline\n\n"
    "pStop   - stops audio pipeline\n\n"
    "midiRec <midi file path> <output name> <synth ID> - reads MIDI file and records it to specified .WAV file using specific synthesizer\n\n"
    "\n"
    "synthSave <load/save> <save file path> <synth ID> - loads or saves synthesizer configuration\n\n"
    "synthAdd - adds new synthesizer and returns its ID\n\n"
    "synthRemove <synth ID> - removes synthesizer by its ID\n\n"
    "synthCount - returns the total count of the synthesizers\n\n"
    "synthConnect <synth ID> <inputID> - connects specified synthesizer with specified input so the synth will receive keyboard state from that input\n\n"
    "synthDisconnect <synth ID> - removes the connection so the synthesizer wont't be used until new data stream is connected\n\n"
    "\n\n"
    "inputAdd <type> <stream path> <key count> <optional: key map file path> - adds new input and returns its ID, where the type is 'keyboard' or 'midi', stream path describes stream location, key count tells how many notes it should use, key map file path (if type is keybard) describes map file location that will provide keyboard layout interpretation\n\n"
    "inputRemove <input ID> - removes input by its I\n\n"
    "inputCount - returns the total count of the inputs\n\n"
    "\n"
    "\n";
    std::printf("%s\n", help.c_str());
}

void SynthUserInterface::commandPipelineStart(){
    if(audioPipeline->start()){
        std::printf("Couldn't start pipeline!\n");
        error = true;
        return;
    }
    if (terminalInput == true){
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
        error = true;
        return;
    }
    if (audioPipeline->isRuning()){
        std::printf("To continue audio pipeline have to be idle (run pStop)\n");
        error = true;
        return;
    }

    short synthID;
    if (numberFromToken(3, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("Given synthesizer ID (%i) is not valid\n", synthID);
        error = true;
        return;
    }

    std::printf("Reading MIDI file: %s\n", inputTokens[1]);
    MIDI::MidiFileReader midiReader(inputTokens[1] , audioPipeline->getAudioInfo()->sampleSize, audioPipeline->getAudioInfo()->sampleRate);

    auto start = std::chrono::high_resolution_clock::now();
    if (audioPipeline->recordUntilStreamEmpty(midiReader, synthID, inputTokens[2])){
        std::printf("Something went wrong!\n");
        error = true;
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
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(3, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("Given synthesizer ID (%i) is not valid\n", synthID);
        error = true;
        return;
    }

    if (std::strcmp("load", inputTokens[1]) == 0){
        if (audioPipeline->loadSynthConfig(inputTokens[2], synthID)){
            std::printf("Something went wrong!\n");
            error = true;
            return;
        }
        std::printf("Configuration loaded\n");
    } else if (std::strcmp("save", inputTokens[1]) == 0){
        if (audioPipeline->saveSynthConfig(inputTokens[2], synthID)){
            std::printf("Something went wrong!\n");
            error = true;
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
    stopPipeline();
    if (inputTokenCount < 2){
        std::printf("Usage: synthRemove <synth ID>\n");
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->removeSynthesizer(synthID)){
        std::printf("Something went wrong!\n");
        error = true;
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
        error = true;
        return;
    }

    AKeyboardRecorder* newInput;
    if (std::strcmp(inputTokens[2], "keyboard") == 0){
        if (inputTokenCount < 5){
            std::printf("Usage: inputAdd keyboard <stream path> <key count> <key map file path>\n");
            error = true;
            return;
        }
        std::printf("Interpreting as keyboard input\n");
        InputMap map(inputTokens[5]);
        ushort mapKeyCount = map.getKeyCount();
        ushort userKeyCount;
        if (numberFromToken(4, userKeyCount)){
            error = true;
            return;
        }
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
        ushort userKeyCount;
        if (numberFromToken(4, userKeyCount)){
            error = true;
            return;
        }
        ushort inputKeyCount = userKeyCount;

        if (keyCount < inputKeyCount){
            std::printf("System configuration does not support specified key count: %d\n system will use: %d\n", userKeyCount, keyCount);
            inputKeyCount = keyCount;
        }
        newInput = new KeyboardRecorder_DevSnd(inputKeyCount);

    } else {
        std::printf("Not known device type: %s\n", inputTokens[2]);
        error = true;
        return;
    }

    newInput->init(inputTokens[3], audioPipeline->getAudioInfo()->sampleSize, audioPipeline->getAudioInfo()->sampleRate);

    short inputID = audioPipeline->addInput(newInput);
    std::printf("Input created with ID: %d\n", inputID);
}

void SynthUserInterface::commandInputRemove(){
    stopPipeline();
    if (inputTokenCount < 2){
        std::printf("Usage: inputRemove <synth ID>\n");
        error = true;
        return;
    }
    short inputID;
    if (numberFromToken(1, inputID)){
        error = true;
        return;
    }
    if (audioPipeline->removeInput(inputID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Input (%d) removed\n", inputID);
}

void SynthUserInterface::commandInputCount(){
    short count = audioPipeline->getInputCount();
    std::printf("Input count: %d\n", count);
}

void SynthUserInterface::commandSynthConnect(){
    stopPipeline();
    if (inputTokenCount < 3){
        std::printf("Usage: synthConnect <synth ID> <inputID>\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    short inputID;
    if (numberFromToken(2, inputID)){
        error = true;
        return;
    }
    if (audioPipeline->connectInputToSynth(inputID, synthID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Connected synth (%d) <- input (%d)\n", synthID, inputID);
}

void SynthUserInterface::commandSynthDisconnect(){
    stopPipeline();
    if (inputTokenCount < 2){
        std::printf("Usage: synthConnect <synth ID>\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->disconnectSynth(synthID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Disconnected synth (%d)\n", synthID);
}

void SynthUserInterface::commandSetOutputBuffer(){
    stopPipeline();
    if (inputTokenCount < 2){
        std::printf("Usage: setOut <ID type> <ID>\n");
        error = true;
        return;
    }
    short ID;
    if (numberFromToken(2, ID)){
        error = true;
        return;
    }
    pipeline::ID_type IDType = pipeline::stringToIDType(inputTokens[1]);

    if (IDType != pipeline::SYNTH && IDType != pipeline::COMP){
        std::printf("ID type invalid: %s\n", inputTokens[1]);
    }
    if (audioPipeline->setOutputBuffer(ID, IDType)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Output set to: %s(%d)\n", inputTokens[1], ID);
}

void SynthUserInterface::commandSynthSettings(){
    if (inputTokenCount < 2){
        std::printf("Usage: synthSettings <synth ID> {optional: setting type}\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("Synth (%d) does not exist\n", synthID);
        error = true;
        return;
    }

    if (inputTokenCount > 2){
        for (uint i = 2; i < inputTokenCount; i++){
            synthesizer::settings_name setting = synthesizer::stringToSettingName(inputTokens[i]);
            if (setting == synthesizer::INVALID){
                if (std::strcmp("type", inputTokens[i]) != 0 && std::strcmp("TYPE", inputTokens[i]) != 0){
                    std::printf("Invalid setting type: %s\n", inputTokens[i]);
                } else {
                    synthesizer::generator_type generatorType = audioPipeline->getSynthType(synthID);
                    switch (generatorType) {
                        case synthesizer::SINE:
                            std::printf("%s: SINE\n", inputTokens[i]);
                            break;
                        case synthesizer::SQUARE:
                            std::printf("%s: SQUARE\n", inputTokens[i]);
                            break;
                        case synthesizer::SAWTOOTH:
                            std::printf("%s: SAWTOOTH\n", inputTokens[i]);
                            break;
                        case synthesizer::TRIANGLE:
                            std::printf("%s: TRIANGLE\n", inputTokens[i]);
                            break;
                        case synthesizer::NOISE1:
                            std::printf("%s: NOISE1\n", inputTokens[i]);
                            break;
                        case synthesizer::INVALID_GEN:
                            std::printf("%s: INVALID_GEN\n", inputTokens[i]);
                            break;
                    }
                }
            } else {
                std::printf("%s: %f\n", inputTokens[i], audioPipeline->getSynthSetting(synthID, setting));
            }
        }

    } else {
        const synthesizer::settings& settings = *audioPipeline->getSynthSettings(synthID);
        synthesizer::generator_type generatorType = audioPipeline->getSynthType(synthID);
        switch (generatorType) {
            case synthesizer::SINE:
                std::printf("TYPE: SINE\n");
                break;
            case synthesizer::SQUARE:
                std::printf("TYPE: SQUARE\n");
                break;
            case synthesizer::SAWTOOTH:
                std::printf("TYPE: SAWTOOTH\n");
                break;
            case synthesizer::TRIANGLE:
                std::printf("TYPE: TRIANGLE\n");
                break;
            case synthesizer::NOISE1:
                std::printf("TYPE: NOISE1\n");
                break;
            case synthesizer::INVALID_GEN:
                std::printf("TYPE: INVALID_GEN\n");
                break;
        }
        std::printf(
        "VOLUME:  %f\n"
        "PITCH:   %d\n"
        "STEREO:  %f\n"
        "ATTACK:  %f\n"
        "SUSTAIN: %f\n"
        "FADE:    %f\n"
        "FADE TO: %f\n"
        "RELEASE: %f\n",
        settings.volume, settings.pitch, settings.stereoMix, settings.attack.raw, settings.sustain.raw, settings.fade.raw, settings.fadeTo, settings.release.raw);
    }
}

void SynthUserInterface::commandSynthModify(){
    if (inputTokenCount < 4 || inputTokenCount%2 == 1){
        std::printf("Usage: synthModify <synth ID> {<setting type> <value>}\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("%d is not valid SYNTH id\n", synthID);
        error = true;
        return;
    }

    for (ushort i = 2; i < inputTokenCount; i+=2){
        if (std::strcmp("type", inputTokens[i]) == 0 || std::strcmp("TYPE", inputTokens[i]) == 0){
            synthesizer::generator_type type = synthesizer::stringToSynthType(inputTokens[i+1]);
            if (type == synthesizer::INVALID_GEN){
                std::printf("Invalid generator type: %s\n", inputTokens[i+1]);
            } else {
                audioPipeline->setSynthSetting(synthID, type);
                std::printf("%s: %s\n", inputTokens[i], inputTokens[i+1]);
            }
        } else {
            synthesizer::settings_name setting = synthesizer::stringToSettingName(inputTokens[i]);
            if (setting == synthesizer::INVALID){
                std::printf("Invalid setting type: %s\n", inputTokens[i]);
            } else {
                float value;
                if (numberFromToken(i+1, value) == 0){
                    audioPipeline->setSynthSetting(synthID, setting, value);
                    std::printf("%s: %f\n", inputTokens[i], value);
                }
            }
        }
    }
}

void SynthUserInterface::commandReinitializeID(){
    stopPipeline();
    audioPipeline->reorganizeIDs();
    std::printf("All IDs were reinitialized - connections will be broken\n");
}

void SynthUserInterface::commandExecuteScript(){
    if (inputTokenCount < 2){
        std::printf("Usage: execute <file path>\n");
        error = true;
        return;
    }
    if (error){
        std::printf("Warning: error flag set - clearing before executing\n");
        clearErrorFlag();
    }
    std::printf("Executing...\n");
    if (scriptReader.executeScript(inputTokens[1], true)){
        std::printf("%s", scriptReader.getLastError().c_str());
        error = true;
        return;
    }
    std::printf("Execution succesfull\n");
}
