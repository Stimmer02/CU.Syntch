#include "SynthUserInterface.h"
#include "Pipeline/IDManager.h"
#include <string>



SynthUserInterface::SynthUserInterface(audioFormatInfo audioInfo, AKeyboardRecorder*& keyboardInput, IKeyboardInput*& userInput, ushort keyCount){
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
    delete userInput;
}

char SynthUserInterface::start(){
    if (userInput == nullptr){
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    audioPipeline->start();
    std::printf("ALL RUNNING\n");
    running = true;

    audioPipeline->pauseInput();
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
        {"exit",   &SynthUserInterface::commandExit},
        {"toggle", &SynthUserInterface::commandToggle},
        {"help",   &SynthUserInterface::commandHelp},
        {"pStart", &SynthUserInterface::commandPipelineStart},
        {"pStop",  &SynthUserInterface::commandPipelineStop},
        {"midiRec",&SynthUserInterface::commandMidiRecord},
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

void SynthUserInterface::commandHelp(){
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

    if(audioPipeline->recordUntilStreamEmpty(midiReader, synthID, inputTokens[2])){
        std::printf("Something went wrong!\n");
        return;
    }
    std::printf("File successfully saved as: %s\n", inputTokens[2]);
}
