#include "SynthUserInterface.h"
#include "Pipeline/ComponentManager.h"
#include "Pipeline/IDManager.h"
#include "enumConversion.h"


using namespace pipeline;

SynthUserInterface::SynthUserInterface(std::string terminalHistoryPath, audioFormatInfo audioInfo, ushort keyCount): scriptReader(this), history(terminalHistoryPath){
    userInput = nullptr;
    this->keyCount = keyCount;

    inputTokens = new const char*[inputTokenMax];
    inputTokenCount = 0;
    initializeCommandMap();

    audioPipeline = new AudioPipelineManager(audioInfo, keyCount);

    terminalInput = true;
    running = false;
    error = false;
    loopDelay = 1000/30;
}


SynthUserInterface::~SynthUserInterface(){
    delete audioPipeline;
    delete[] inputTokens;
    delete commandMap;
    if (userInput != nullptr){
        delete userInput;
    }
}

char SynthUserInterface::setUserInput(IKeyboardInput*& userInput){
    this->userInput = userInput;
    userInput = nullptr;

    if (this->userInput->start()){
        delete this->userInput;
        this->userInput = nullptr;
        terminalInput = false;
        return 1;
    }

    return 0;
}


char SynthUserInterface::start(){
    running  = true;
    while (this->running){
        readInput();
        parseInput();
    }

    audioPipeline->stop();
    if (userInput != nullptr){
        userInput->stop();
    }
    std::printf("ALL STOPPED\n");
    return 0;
}

void SynthUserInterface::browseHistory(){
    if (userInput == nullptr){
        printf("User input is not set, run setUserInput\n");
        error = true;
        return;
    }
    terminalDiscard.disableInput();
    std::printf("\e[A\e[2K\e[G\e[30m\e[107mH⮞\e[30m");
    std::string entry = history.getPreviousEntry();
    std::printf("%s", entry.c_str());
    fflush(stdout);
    waitUntilKeyReleased(KEY_ENTER);

    bool specialSequence = true;
    while (specialSequence){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        if (userInput->getKeyState(KEY_UP)){
            entry = history.getPreviousEntry();
            std::printf("e[0m\e[2K\e[G\e[30m\e[107mH⮞\e[30m %s\e[0m", entry.c_str());
            fflush(stdout);
            waitUntilKeyReleased(KEY_UP);
        } else if (userInput->getKeyState(KEY_DOWN)){
            entry = history.getNextEntry();
            std::printf("e[0m\e[2K\e[G\e[30m\e[107mH⮞\e[30m %s\e[0m", entry.c_str());
            fflush(stdout);
            waitUntilKeyReleased(KEY_DOWN);
        } else if (userInput->getKeyState(KEY_ENTER)){
            std::printf("\n");
            inputLine = entry;
            parseInput();
            std::printf("\e[30m\e[107mH⮞\e[30m %s\e[0m", entry.c_str());
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

    if (inputLine.length() == 0){
        return;
    }

    if (inputLine[0] == ' ' || inputLine[0] == '#'){
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
        } else if (inputLine[i] == '#'){
            return;
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
        } else if (inputLine[i] == '#'){
            inputLine[i] = '\0';
            break;
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

std::string SynthUserInterface::concatenateTokens(short startTokenIndex){
    std::string result = inputTokens[startTokenIndex];
    for (short i = startTokenIndex + 1; i < inputTokenCount; i++){
        result += " ";
        result += inputTokens[i];
    }
    return result;
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
        {"help",         &SynthUserInterface::commandHelp},
        {"exit",         &SynthUserInterface::commandExit},
        {"setUserInput", &SynthUserInterface::commandSetUserInput},
        {"toggle",       &SynthUserInterface::commandToggle},
        {"pStart",       &SynthUserInterface::commandPipelineStart},
        {"pStop",        &SynthUserInterface::commandPipelineStop},
        {"setOut",       &SynthUserInterface::commandSetOutputBuffer},
        {"execute",      &SynthUserInterface::commandExecuteScript},
        {"midiRec",      &SynthUserInterface::commandMidiRecord},
        {"clear",        &SynthUserInterface::commandClear},
        {"system",       &SynthUserInterface::commandSystem},
        {"idReinit",     &SynthUserInterface::commandReinitializeID},

        {"\e[A",         &SynthUserInterface::browseHistory},

        {"visGet",  &SynthUserInterface::commandVisualizerSettings},
        {"visSet",  &SynthUserInterface::commandVisualizerModify},

        {"synthAdd",        &SynthUserInterface::commandSynthAdd},
        {"synthRemove",     &SynthUserInterface::commandSynthRemove},
        {"synthConnect",    &SynthUserInterface::commandSynthConnect},
        {"synthDisconnect", &SynthUserInterface::commandSynthDisconnect},
        {"synthCount",      &SynthUserInterface::commandSynthCount},
        {"synthGet",        &SynthUserInterface::commandSynthSettings},
        {"synthSet",        &SynthUserInterface::commandSynthModify},
        {"synthInfo",       &SynthUserInterface::commandSynthInfo},
        {"synthSave",       &SynthUserInterface::commandSynthSave},
        {"synthTypes",      &SynthUserInterface::commandSynthTypes},

        {"inputAdd",    &SynthUserInterface::commandInputAdd},
        {"inputRemove", &SynthUserInterface::commandInputRemove},
        {"inputCount",  &SynthUserInterface::commandInputCount},

        {"midiAdd",    &SynthUserInterface::commandMidiReaderAdd},
        {"midiSet",    &SynthUserInterface::commandMidiReaderSet},
        {"midiPlay",   &SynthUserInterface::commandMidiReaderPlay},
        {"midiPause",  &SynthUserInterface::commandMidiReaderPause},
        {"midiRecord", &SynthUserInterface::commandMidiReaderRecord},
        {"midiRewind", &SynthUserInterface::commandMidiReaderRewind},
        {"midiList",   &SynthUserInterface::commandMidiReaderList},

        {"compAdd",        &SynthUserInterface::commandComponentAdd},
        {"compRemove",     &SynthUserInterface::commandComponentRemove},
        {"compConnect",    &SynthUserInterface::commandComponentConnect},
        {"compDisconnect", &SynthUserInterface::commandComponentDisconnect},
        {"compConnection", &SynthUserInterface::commandComponentGetConnection},
        {"compCount",      &SynthUserInterface::commandComponentCount},
        {"compGet",        &SynthUserInterface::commandComponentSettings},
        {"compSet",        &SynthUserInterface::commandComponentModify},
        {"compTypes",      &SynthUserInterface::commandComponentTypes},

        {"aCompConnect",    &SynthUserInterface::commandAdvComponentConnect},
        {"aCompDisconnect", &SynthUserInterface::commandAdvComponentDisconnect},
        {"aCompInfo",       &SynthUserInterface::commandAdvComponentInfo},

        {"statRecord", &SynthUserInterface::commandRecordStatistics},
        {"statStop",   &SynthUserInterface::commandStopRecordStatistics}
    };
}

void SynthUserInterface::commandExit(){
    running = false;
    std::printf("System shuting down...\n");
}

void SynthUserInterface::commandToggle(){
    if (userInput == nullptr){
        printf("User input is not set, run setUserInput\n");
        error = true;
        return;
    }
    if (audioPipeline->isRuning() == false){
        std::printf("Pipeline is not running\n");
        error = true;
        return;
    }
    terminalDiscard.disableInput();
    audioPipeline->reausumeInput();
    terminalInput = false;
    std::printf("Terminal input disabled, to enable it press \"Ctrl+Q\"\n");
    if (audioPipeline->isUsingVisualizer()){
        audioPipeline->startVisualizer();
        std::printf("Visualizer is printing\n");
    }
    while (terminalInput == false){
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
        if (userInput->getKeyState(KEY_LEFTCTRL) && userInput->getKeyState(KEY_Q)){
            waitUntilKeyReleased(KEY_Q);
            terminalInput = true;
        }
    }
    if (audioPipeline->isUsingVisualizer()){
        audioPipeline->stopVisualizer();
    }
    audioPipeline->pauseInput();
    terminalDiscard.enableInput();
    std::printf("Terminal input enabled\n");
}

void SynthUserInterface::commandHelp(){

    static const std::string list =
    "COMMAND LIST:\n"
    "\n"
    "SYSTEM\n"
    "   help\n"
    "   exit\n"
    "   setUserInput\n"
    "   toggle\n"
    "   pStart\n"
    "   pStop\n"
    "   setOut\n"
    "   execute\n"
    "   midiRec\n"
    "   clear\n"
    "   system\n"
    "   idReinit\n"
    "\n"
    "VISUALIZER\n"
    "   visGet\n"
    "   visSet\n"
    "   visKey\n"
    "\n"
    "SYNTHESIZER\n"
    "   synthAdd\n"
    "   synthRemove\n"
    "   synthConnect\n"
    "   synthDisconnect\n"
    "   synthCount\n"
    "   synthGet\n"
    "   synthSet\n"
    "   synthInfo\n"
    "   synthSave\n"
    "\n"
    "INPUT\n"
    "   inputAdd\n"
    "   inputRemove\n"
    "   inputCount\n"
    "\n"
    "MIDI\n"
    "   midiAdd\n"
    "   midiSet\n"
    "   midiPlay\n"
    "   midiPause\n"
    "   midiRecord\n"
    "   midiRewind\n"
    "   midiList\n"
    "\n"
    "COMPONENT\n"
    "   compAdd\n"
    "   compRemove\n"
    "   compConnect\n"
    "   compDisconnect\n"
    "   compConnection\n"
    "   compCount\n"
    "   compGet\n"
    "   compSet\n"
    "   compTypes\n"
    "\n"
    "ADVANCED COMPONENT\n"
    "   aCompConnect\n"
    "   aCompDisconnect\n"
    "   aCompInfo\n"
    "\n"
    "STATISTICS\n"
    "   statRecord\n"
    "   statStop\n"
    "\n";

    static const std::string help =
    "HELP PROMPT:\n"
    "Usage: command <arguments>\n"
    "\n"
    "SYSTEM - managment and utilities\n"
    "   help    - shows this message, argument \"list\" will show no descriptions\n"
    "   exit    - exits the program (if program does not turn off: press any key on every previously connected device to end reading threads)\n"
    "   setUserInput <keyboard system stream> - sets keyboard stream to be used for receiving keyboard combinations (eg. Ctrl+Q)\n"
    "   toggle  - toggles input between synthesizer and console, after switching to synthesizer press Ctrl+Q to switch back\n"
    "   pStart  <v> - starts audio pipeline. Use with \"v\" argument to start with audio spectrum visualizer (only after invoking \"toggle\" command it will be shown)\n"
    "   pStop   - stops audio pipeline\n"
    "   setOut  <ID type> <ID> - sets synthesize or advanced component to be system audio output\n"
    "   execute <file path> - executes script of given path\n"
    "   midiRec <midi file path> <output name> <synth ID> - [DEPRECATED] reads MIDI file and records it to specified .WAV file using specific synthesizer\n"
    "   clear   - clears console\n"
    "   system  {system command and its arguments} - executes system command in current shell\n"
    "   idReinit - reinitializes all ID's in the order of components memory allocation, this way there are no gaps between them and also breaks current initialization (used for clearing system)\n"
    "\n"
    "VISUALIZER - audio spectrum visualizer (to enable run \"pStart v\")\n"
    "   visGet  - returns visualizer settings\n"
    "   visSet  {<setting name> <value>} - sets specified setting values\n"
    "\n"
    "SYNTHESIZER - audio signall creation\n"
    "   synthAdd     - adds new synthesizer and returns its ID\n"
    "   synthRemove  <synth ID> - removes synthesizer by its ID\n"
    "   synthConnect <synth ID> <inputID> - connects specified synthesizer with specified input so the synth will receive keyboard state from that input\n"
    "   synthDisconnect <synth ID> - removes the connection so the synthesizer wont't be used until new data stream is connected\n"
    "   synthCount   - returns the total count of the synthesizers\n"
    "   synthGet     <synth ID> {optional: <setting name>} - returns synthesizer settings (all/specified)\n"
    "   synthSet     <synth ID> {<setting name> <value>} - sets specified setting values\n"
    "   synthInfo    - returns information about queue of specified synthesizer\n"
    "   synthSave    <synth ID> <load/save> <save file path> - loads or saves synthesizer configuration\n"
    "   synthTypes   - returns a list of all avaliable synthesizer types\n"
    "\n"
    "INPUT - midi input\n"
    "   inputAdd <type> <stream path> <key count> <optional: key map file path> - adds new input and returns its ID, where the type is 'keyboard' or 'midi', stream path describes stream location, key count tells how many notes it should use, key map file path (if type is keybard) describes map file location that will provide keyboard layout interpretation\n"
    "   inputRemove <input ID> - removes input by its I\n"
    "   inputCount - returns the total count of the inputs\n"
    "\n"
    "MIDI - midi file playback (other MIDI functions work the same as INPUT). May not work on all MIDI files due to not implementing whole MIDI functionality\n"
    "   midiAdd     - adds new MIDI file reader to the system with INPUT ID\n"
    "   midiSet     <file path> - sets MIDI file reader a file to read\n"
    "   midiPlay    optional: {input ID} - starts playback of specified MIDI reader or all MIDI readers\n"
    "   midiPause   optional: {input ID} - pauses playback of specified MIDI reader or all MIDI readers\n"
    "   midiRecord  <output name> optional: {flags} - records MIDI file to .WAV file using all MIDI readers\n"
    "      offline - records without playback, as fast as possible\n"
    "      user    - toggles user input then starts recording\n"
    "      time <file to append> - saves elapsed time to specified file\n"
    "   midiRewind  optional: {input ID} - rewinds playback of specified MIDI reader or all MIDI readers\n"
    "   midiList    - returns list of all MIDI readers\n"
    "\n"
    "COMPONENT - audio stream manipulation\n"
    "   compAdd      <component type> - adds specified component or advanced component to the systen\n"
    "   compRemove   <component ID> - removes specified component\n"
    "   compConnect  <component type> <parentType> <ID> - connects specified component to synthesizer of advanced component\n"
    "   compDisconnect <component ID> - disconnects component from synthesizer of advanced component\n"
    "   compConnection <component ID> - returns infornation about what system element specified component is connected to\n"
    "   compCount    - returns total count of components in the system\n"
    "   compGet      <component ID> {optional: <setting name>} - returns component settings (all/specified)\n"
    "   compSet      <component ID> {<setting name> <value>} - sets specified setting values\n"
    "   compTypes    - returns a list of all avaliable compomnent types\n"
    "\n"
    "ADVANCED COMPONENT - advanced audio stream manipulation\n"
    "   aCompConnect    <component ID> <input index> <ID type> <ID> - connects advanced component input (specified by \"input index\") to another advanced component or synthesizer, extending pipeline, possibly creating branches or merging them\n"
    "   aCompDisconnect <component ID> <input index> - disconnect advanced component input from source\n"
    "   aCompInfo       <component ID> - returns information about queue of specified advanced component\n"
    "\n"
    "STATISTICS\n"
    "   statRecord <update time interval> <file path>- records statistics to specified file with specified update time interval in seconds\n"
    "   statStop - stops recording statistics\n"
    "\n";


    if (inputTokenCount > 1 && std::strcmp("list", inputTokens[1]) == 0){
        std::printf("%s\n", list.c_str());
    } else {
        std::printf("%s\n", help.c_str());
    }
}

void SynthUserInterface::commandComponentTypes(){
    const char* types =
    "VOLUME:\n"
    "   volume - changes the volume of a signall\n"
    "\n"
    "PAN:\n"
    "   pan - balances stereo effect between left and right\n"
    "\n"
    "COMPRESSOR:\n"
    "   threshold - volume after whitch compression will be applied\n"
    "   ratio     - compression ratio for each step above the threshold\n"
    "   step      - value after whitch ratio will be fully applied (effect stacks with each step)\n"
    "   attack    - time in seconds to fully register volume level increase\n"
    "   release   - time in seconds to fully register volume level decrease\n"
    "   vol       - output volume multiplication\n"
    "\n"
    "ECHO:\n"
    "   rvol    - volume of the right echo channel\n"
    "   lvol    - volume of the left echo channel\n"
    "   delay   - delay in seconds between repeats\n"
    "   fade    - volume multiplication between repeats\n"
    "   repeats - count of repeats\n"
    "\n"
    "DISTORTION:\n"
    "   gain     - input volume multiplication\n"
    "   compress - threshold that will cut everything above it's value\n"
    "   symmetry - positive values will increase positive phase gain, negative - negative phase gain\n"
    "   vol      - output volume multiplication\n"
    "\n"
    "DESTROY:\n"
    "   \e[30;41m[MAY CAUSE HEARING OR SPEAKERS DAMAGE IF USED IMPROPERLY! - CAN DRASTICALLY INCREASE VOLUME LEVEL]\e[0m\n"
    "   subtract - value that will be substracted from positive phase and added to negative phase\n"
    "\n"
    "ADVANCED:\n"
    "SUM2:\n"
    "   vol0 - changes the volume of a firts input signall\n"
    "   vol1 - changes the volume of a second input signall\n"
    "\n"
    "   in[0] - first signall to concatenate\n"
    "   in[1] - second signall to concatenate\n"
    "\n"
    "\n"
    "COPY:\n"
    "   vol - changes the volume of input signall\n"
    "\n"
    "   in[0] - signall to copy\n"
    "\n";

    std::printf("%s", types);
}

void SynthUserInterface::commandPipelineStart(){
    if (audioPipeline->isRuning()){
        std::printf("Pipeline is already running\n");
        error = true;
        return;
    }
    if (inputTokenCount > 1 && std::strcmp("v", inputTokens[1]) == 0){
        if (audioPipeline->start(true)){
            std::printf("Couldn't start pipeline!\n");
            error = true;
            return;
        }
        std::printf("Visualizer enabled\n");
    } else {
        if (audioPipeline->start(false)){
            std::printf("Couldn't start pipeline!\n");
            error = true;
            return;
        }
    }
    
    uint waitCounter = 0;
    while (audioPipeline->isRuning() == false){
        if (waitCounter > 1000/loopDelay + 1){;
            audioPipeline->stop();
            std::printf("Couldn't start pipeline on time!\n");
            error = true;
            return;
        }
        waitCounter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(loopDelay));
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

void SynthUserInterface::commandMidiRecord(){//DEPRECATED
    if (inputTokenCount < 4){
        std::printf("[DEPRECATED] Usage: midiRec <midi file path> <output name> <synth ID>\n");
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

void SynthUserInterface::commandVisualizerModify(){
    if (inputTokenCount < 3 || inputTokenCount % 2 == 0){
        std::printf("Usage: visSet {<setting name> <value>}\n");
        error = true;
        return;
    }
    for (short i = 1; i < inputTokenCount; i += 2){
        if (std::strcmp(inputTokens[i], "fps") == 0){
            float fps;
            if (numberFromToken(i + 1, fps)){
                error = true;
                return;
            }
            fps = audioPipeline->setVisualizerFps(fps);
            std::printf("FPS set to: %.2f\n", fps);
        } else if (std::strcmp(inputTokens[i], "samples") == 0){
            uint samples;
            if (numberFromToken(i + 1, samples)){
                error = true;
                return;
            }
            audioPipeline->setVisualizerWindowSize(samples);
            std::printf("Samples set to: %i\n", audioPipeline->getVisualizerWindowSize());
        } else if (std::strcmp(inputTokens[i], "low") == 0){
            float low;
            if (numberFromToken(i + 1, low)){
                error = true;
                return;
            }
            audioPipeline->setVisualizerLowScope(low);
            std::printf("Low set to: %.2f\n", audioPipeline->getVisualizerLowScope());
        } else if (std::strcmp(inputTokens[i], "high") == 0){
            float high;
            if (numberFromToken(i + 1, high)){
                error = true;
                return;
            }
            audioPipeline->setVisualizerHighScope(high);
            std::printf("High set to: %.2f\n", audioPipeline->getVisualizerHighScope());
        } else if (std::strcmp(inputTokens[i], "volume") == 0){
            float volume;
            if (numberFromToken(i + 1, volume)){
                error = true;
                return;
            }
            audioPipeline->setVisualizerVolume(volume);
            std::printf("Volume set to: %.2f\n", audioPipeline->getVisualizerVolume());
        } else {
            std::printf("Unknown setting: %s\n", inputTokens[i]);
            error = true;
            return;
        }
    }

    
}

void SynthUserInterface::commandVisualizerSettings(){
    std::printf(
        "VISUALIZER SETTINGS:\n"
        "   state: %s\n"
        "   fps: %.2f\n"
        "   samples: %i\n"
        "   low: %.2fHz\n"
        "   high: %.2fHz\n"
        "   volume: %.2f\n",
        audioPipeline->isUsingVisualizer() ? "enabled" : "disabled",
        audioPipeline->getVisualizerFps(),
        audioPipeline->getVisualizerWindowSize(),
        audioPipeline->getVisualizerLowScope(),
        audioPipeline->getVisualizerHighScope(),
        audioPipeline->getVisualizerVolume()
    );
}


void SynthUserInterface::commandSynthSave(){
    if (inputTokenCount < 4){
        std::printf("Usage: synthSave <synth ID> <load/save> <save file path>\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("SYNTH(%d) does not exist\n", synthID);
        error = true;
        return;
    }

    if (std::strcmp("load", inputTokens[2]) == 0){
        if (audioPipeline->loadSynthConfig(inputTokens[3], synthID)){
            std::printf("Something went wrong!\n");
            error = true;
            return;
        }
        std::printf("Configuration loaded\n");
    } else if (std::strcmp("save", inputTokens[2]) == 0){
        if (audioPipeline->saveSynthConfig(inputTokens[3], synthID)){
            std::printf("Something went wrong!\n");
            error = true;
            return;
        }
        std::printf("Configuration saved\n");
    } else {
        std::printf("Unknown option: %s\n", inputTokens[2]);
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
    std::printf("SYNTH(%d) removed\n", synthID);
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
    if (std::strcmp(inputTokens[1], "keyboard") == 0){
        if (inputTokenCount < 5){
            std::printf("Usage: inputAdd keyboard <stream path> <key count> <key map file path>\n");
            error = true;
            return;
        }
        std::printf("Interpreting as keyboard input\n");
        InputMap* map =  new InputMap(inputTokens[4]);
        ushort inputKeyCount;
        if (numberFromToken(3, inputKeyCount)){
            error = true;
            return;
        }
        if (keyCount < inputKeyCount){
            std::printf("System configuration does not support specified key count: %d\n system will use: %d\n", inputKeyCount, keyCount);
            inputKeyCount = keyCount;
        }
        //
        // ushort mapKeyCount = map->getKeyCount();
        // if (mapKeyCount < inputKeyCount){
        //     std::printf("Provided key map does not support specified key count: %d\n system will use: %d\n", userKeyCount, mapKeyCount);
        //     inputKeyCount = mapKeyCount;
        // }
        newInput = new KeyboardRecorder_DevInput(inputKeyCount, map);

    } else if (std::strcmp(inputTokens[1], "midi") == 0){
        std::printf("Interpreting as midi input\n");
        ushort inputKeyCount;
        if (numberFromToken(3, inputKeyCount)){
            error = true;
            return;
        }
        if (keyCount < inputKeyCount){
            std::printf("System configuration does not support specified key count: %d\n system will use: %d\n", inputKeyCount, keyCount);
            inputKeyCount = keyCount;
        }
        newInput = new KeyboardRecorder_DevSnd(inputKeyCount);

    } else {
        std::printf("Not known device type: %s\n", inputTokens[1]);
        error = true;
        return;
    }

    if (newInput->init(inputTokens[2], audioPipeline->getAudioInfo()->sampleSize, audioPipeline->getAudioInfo()->sampleRate)){
        std::printf("Could not initialize new input from stream: %s\n", inputTokens[2]);
        error = true;
        delete newInput;
        return;
    }

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
    std::printf("INPUT(%d) removed\n", inputID);
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
    std::printf("Synth connected SYNTH(%d) <- INPUT(%d)\n", synthID, inputID);
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
    std::printf("SYNTH(%d) disconnected\n", synthID);
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
        std::printf("Usage: synthSettings <synth ID> {optional: setting name}\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("SYNTH(%d) does not exist\n", synthID);
        error = true;
        return;
    }

    if (inputTokenCount > 2){
        for (uint i = 2; i < inputTokenCount; i++){
            synthesizer::settings_name setting = synthesizer::stringToSettingName(inputTokens[i]);
            if (setting == synthesizer::INVALID){
                if (std::strcmp("type", inputTokens[i]) != 0 && std::strcmp("TYPE", inputTokens[i]) != 0){
                    std::printf("Invalid setting name: %s\n", inputTokens[i]);
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
        const synthesizer::settings_CUDA& settings = *audioPipeline->getSynthSettings(synthID);
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
        std::printf("Usage: synthSet <synth ID> {<setting name> <value>}\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->IDValid(pipeline::SYNTH, synthID) == false){
        std::printf("SYNTH(%d) does not exist\n", synthID);
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
                std::printf("Invalid setting name: %s\n", inputTokens[i]);
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

void SynthUserInterface::commandSynthInfo(){
     if (inputTokenCount < 2){
        std::printf("Usage: synthInfo <synth ID>\n");
        error = true;
        return;
    }
    short synthID;
    if (numberFromToken(1, synthID)){
        error = true;
        return;
    }
    if (audioPipeline->printSynthInfo(synthID)){
        std::printf("SYNTH(%d) does not exist\n", synthID);
        error = true;
        return;
    }
}

void SynthUserInterface::commandSynthTypes(){
    const char* types =
    "SYNTH TYPES:\n"
    "SINE:\n"
    "   Simple sine wave generator\n"
    "\n"
    "SQUARE:\n"
    "   Simple square wave generator\n"
    "\n"
    "SAWTOOTH:\n"
    "   Simple sawtooth wave generator\n"
    "\n"
    "TRIANGLE:\n"
    "   Simple triangle wave generator\n"
    "\n"
    "NOISE1:\n"
    "   Non musical noise generator\n";

    std::printf("%s", types);
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

void SynthUserInterface::commandSystem(){
    if (inputTokenCount < 2){
        std::printf("Usage: system {system command and its arguments}\n");
        error = true;
        return;
    }
    std::string systemCommand = history.getPreviousEntry();
    history.getNextEntry();
    systemCommand.erase(0, 7);
    std::system(systemCommand.c_str());
}

void SynthUserInterface::commandSetUserInput(){
    if (inputTokenCount < 2){
        std::printf("Usage: setUserInput <keyboard system stream>\n");
        error = true;
        return;
    }
    if (this->userInput != nullptr){
        std::printf("User input was already set, removing old one\n");
        this->userInput->stop();
        delete this->userInput;
    }
    IKeyboardInput* userInput = new KeyboardInput_DevInput();
    if (userInput->init(inputTokens[1])){
        std::printf("Could not initialize user input from stream: %s\n", inputTokens[1]);
        delete userInput;
        error = true;
        return;
    }
    if (setUserInput(userInput)){
        std::printf("Could not start new user input\n");
        error = true;
        return;
    }
    std::printf("New user input set and running\n");
}

void SynthUserInterface::commandComponentAdd(){
    if (inputTokenCount < 2){
        std::printf("Usage: compAdd <component type>\n");
        error = true;
        return;
    }

    short newComponentID;

    pipeline::component_type compType = pipeline::stringToComponentType(inputTokens[1]);
    if (compType == pipeline::COMP_INVALID){
        pipeline::advanced_component_type advCompType = pipeline::stringToAdvComponentType(inputTokens[1]);
        if (advCompType == pipeline::ACOMP_INVALID){
            std::printf("Component type does not exist: %s\n", inputTokens[1]);
            error = true;
            return;
        } else {
            newComponentID = audioPipeline->addComponent(advCompType);
        }
    } else {
        newComponentID = audioPipeline->addComponent(compType);
    }


    std::printf("Component created with ID: %d\n", newComponentID);
}

void SynthUserInterface::commandComponentRemove(){
    if (inputTokenCount < 2){
        std::printf("Usage: compRemove <component ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }
    // ID_type IDType;
    // short parentID;
    // audioPipeline->getComponentConnection(componentID, IDType, parentID);
    // if (IDType != pipeline::INVALID){
    //     stopPipeline();
    // }

    if (audioPipeline->isAdvancedComponent(componentID)){
        stopPipeline();
    }

    if (audioPipeline->removeComponent(componentID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("COMP(%d) removed\n", componentID);
}

void SynthUserInterface::commandComponentCount(){
    short count = audioPipeline->getComponentCout();
    std::printf("Component count: %d\n", count);
}

void SynthUserInterface::commandComponentConnect(){
    if (inputTokenCount < 4){
        std::printf("Usage: compConnect <component ID> <parentType> <ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }
    pipeline::ID_type IDType = pipeline::stringToIDType(inputTokens[2]);
    if (IDType != pipeline::SYNTH && IDType != pipeline::COMP){
        std::printf("ID type invalid: %s\n", inputTokens[2]);
    }
    short parentID;
    if (numberFromToken(3, parentID)){
        error = true;
        return;
    }

    if (audioPipeline->connectComponent(componentID, IDType, parentID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }

    std::printf("Component connected: COMP(%d) <- %s(%d)\n", componentID, inputTokens[2], parentID);
}

void SynthUserInterface::commandComponentDisconnect(){
    if (inputTokenCount < 2){
        std::printf("Usage: compDisconnect <component ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }

    if (audioPipeline->disconnectCommponent(componentID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }

    std::printf("COMP(%d) disconnected\n", componentID);
}

void SynthUserInterface::commandComponentGetConnection(){//TODO update
    if (inputTokenCount < 2){
        std::printf("Usage: compConnection <component ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }

    ID_type IDType;
    short parentID;

    if (audioPipeline->getComponentConnection(componentID, IDType, parentID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }

    if (IDType == pipeline::INVALID){
        std::printf("COMP(%d) is not connected\n", componentID);
    } else {
        std::string IDTypeString = pipeline::IDTypeToString(IDType);
        std::printf("COMP(%d) connected to: %s(%d)\n", componentID, IDTypeString.c_str(), parentID);
    }
}

void SynthUserInterface::commandComponentModify(){
    if (inputTokenCount < 4 || (inputTokenCount & 1)){
        std::printf("Usage: compSet <component ID> {<setting name> <value>}\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }

    const componentSettings_CUDA* settings = audioPipeline->getComopnentSettings(componentID);
    if (settings == nullptr){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    for (uint i = 2; i < inputTokenCount; i+=2){
        bool settingFound = false;
        uint j;
        for (j = 0; j < settings->count; j++){
            if (std::strcmp(inputTokens[i], settings->names[j].c_str()) == 0){
                settingFound = true;
                break;
            }
        }
        if (settingFound){
            float settingValue;
            if (numberFromToken(i + 1, settingValue)){
                error = true;
                return;
            }
            if (audioPipeline->setComponentSetting(componentID, j, settingValue)){
                std::printf("Could not set setting %s: %f\n", inputTokens[i], settingValue);
            } else {
                const componentSettings_CUDA* settings = audioPipeline->getComopnentSettings(componentID);
                if (settings == nullptr){
                    std::printf("Something went wrong!\n");
                    error = true;
                    return;
                }
                for (uint j = 0; j < settings->count; j++){
                    if (std::strcmp(inputTokens[i], settings->names[j].c_str()) == 0){
                        settingValue = settings->values[j];
                        break;
                    }
                }
                std::printf("%s: %f\n", inputTokens[i], settingValue);
            }
        } else {
            std::printf("COMP(%d) does not have setting type: %s\n", componentID, inputTokens[i]);
        }
    }
}

void SynthUserInterface::commandComponentSettings(){
    if (inputTokenCount < 2){
        std::printf("Usage: compGet <component ID> {optional :<setting name>}\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }

    const componentSettings_CUDA* settings = audioPipeline->getComopnentSettings(componentID);
    if (settings == nullptr){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    if (inputTokenCount == 2){
        for (uint i = 0; i < settings->count; i++){
            std::printf("%s: %f\n", settings->names[i].c_str(), settings->values[i]);
        }
    } else {
        for (uint i = 2; i < inputTokenCount; i++){
            bool settingFound = false;
            uint j;
            for (j = 0; j < settings->count; j++){
                if (std::strcmp(inputTokens[i], settings->names[j].c_str()) == 0){
                    settingFound = true;
                    break;
                }
            }
            if (settingFound){
                std::printf("%s: %f\n", inputTokens[i], settings->values[j]);
            } else {
                std::printf("COMP(%d) does not have setting type: %s\n", componentID, inputTokens[i]);
            }
        }
    }
}

void SynthUserInterface::commandAdvComponentConnect(){
    stopPipeline();
    if (inputTokenCount < 5){
        std::printf("Usage: aCompConnect <component ID> <input index> <ID type> <ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }
    int index;
    if (numberFromToken(2, index)){
        error = true;
        return;
    }
    ID_type IDType = pipeline::stringToIDType(inputTokens[3]);
    if (IDType != pipeline::SYNTH && IDType != pipeline::COMP){
        std::printf("ID type invalid: %s\n", inputTokens[3]);
    }
    short ID;
    if (numberFromToken(4, ID)){
        error = true;
        return;
    }

    if (audioPipeline->setAdvancedComponentInput(componentID, index, IDType, ID)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Advanced component connected: COMP(%d)[%d] <- %s(%d)\n", componentID, index, inputTokens[3], ID);
}

void SynthUserInterface::commandAdvComponentDisconnect(){
    stopPipeline();
    if (inputTokenCount < 3){
        std::printf("Usage: aCompDisconnect <component ID> <input index>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }
    int index;
    if (numberFromToken(2, index)){
        error = true;
        return;
    }

    if (audioPipeline->tryDisconnectAdvancedCommponent(componentID, index)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Advanced component: COMP(%d)[%d] is set to empty\n", componentID, index);
}

void SynthUserInterface::commandAdvComponentInfo(){
    if (inputTokenCount < 2){
        std::printf("Usage: aCompInfo <component ID>\n");
        error = true;
        return;
    }
    short componentID;
    if (numberFromToken(1, componentID)){
        error = true;
        return;
    }
    if (audioPipeline->printAdvancedComponentInfo(componentID)){
        std::printf("(%d) is not an advanced component\n", componentID);
        error = true;
        return;
    }
}

void SynthUserInterface::commandMidiReaderAdd(){
    // audioPipeline->stop();
    short inputID = audioPipeline->addMidiReader();
    std::printf("MIDI reader created INPUT(%d)\n", inputID);
}

void SynthUserInterface::commandMidiReaderSet(){
    if (inputTokenCount < 3){
        std::printf("Usage: midiSet <input ID> <file path>\n");
        error = true;
        return;
    }
    short inputID;
    if (numberFromToken(1, inputID)){
        error = true;
        return;
    }
    if (audioPipeline->setMidiReader(inputID, concatenateTokens(2))){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("MIDI reader set to: %s\n", concatenateTokens(2).c_str());
}

void SynthUserInterface::commandMidiReaderPlay(){
    if (inputTokenCount > 1){
        short inputID;
        for (uint i = 1; i < inputTokenCount; i++){
            if (numberFromToken(i, inputID)){
                error = true;
                return;
            }
            if (audioPipeline->isMidiReader(inputID) == false){
                std::printf("MIDI INPUT(%d) does not exist\n", inputID);
                continue;
            }
            if (audioPipeline->playMidiReader(inputID)){
                std::printf("Something went wrong!\n");
                error = true;
                return;
            }
            std::printf("MIDI INPUT(%d) playing\n", inputID);
        }
    } else {
        audioPipeline->playMidiReaders();
        std::printf("All MIDI readers playing\n");
    }
}

void SynthUserInterface::commandMidiReaderPause(){
    if (inputTokenCount > 1){
        short inputID;
        for (uint i = 1; i < inputTokenCount; i++){
            if (numberFromToken(i, inputID)){
                error = true;
                return;
            }
            if (audioPipeline->isMidiReader(inputID) == false){
                std::printf("MIDI INPUT(%d) does not exist\n", inputID);
                continue;
            }
            if (audioPipeline->pauseMidiReader(inputID)){
                std::printf("Something went wrong!\n");
                error = true;
                return;
            }
            std::printf("MIDI INPUT(%d) paused\n", inputID);
        }
    } else {
        audioPipeline->pauseMidiReaders();
        std::printf("All MIDI readers paused\n");
    }
}

void SynthUserInterface::commandMidiReaderRecord(){
    if (inputTokenCount < 2){
        std::printf("Usage: midiRecord <output file> optional: {flags}\n");
        error = true;
        return;
    }
    
    bool offlineFlag = false;
    bool userFlag = false;
    bool timeFlag = false;
    std::string saveFilePath;
    for (uint i = 2; i < inputTokenCount; i++){
        if (std::strcmp("offline", inputTokens[i]) == 0){
            offlineFlag = true;
        } else if (std::strcmp("user", inputTokens[i]) == 0){
            userFlag = true;
        } else if (std::strcmp("time", inputTokens[i]) == 0){
            timeFlag = true;
            i++;
            if (i >= inputTokenCount){
                std::printf("Time flag requires time output file\n");
                error = true;
                return;
            }
            saveFilePath = inputTokens[i];            
        } else {
            std::printf("Unrecognised flag: %s\n", inputTokens[i]);
        }
    }
    if (offlineFlag && userFlag){
        std::printf("Flags 'offline' and 'user' are mutually exclusive\n");
        error = true;
        return;
    }
    auto timeStart = std::chrono::system_clock::now();
    std::fstream outputFile;
    if (timeFlag){
        outputFile.open(saveFilePath, std::ios::out);
        if (outputFile.is_open() == false){
            std::printf("Could not open file: %s\n", saveFilePath.c_str());
            error = true;
            return;
        }
    }
    if (offlineFlag){
        if (audioPipeline->isRuning()){
            std::printf("To continue audio pipeline have to be idle (run pStop)\n");
            error = true;
            return;
        }
        double calculationTime;
        if (audioPipeline->recordMidiFilesOffline(inputTokens[1], calculationTime)){
            std::printf("Something went wrong!\n");
            error = true;
            return;
        }
        std::printf("Calculations time: %fs\n", calculationTime);
        if (timeFlag){
            outputFile << calculationTime;
            outputFile.close();
        }
    } else {
        if (audioPipeline->recordMidiFiles(inputTokens[1])){
            std::printf("Something went wrong!\n");
            error = true;
            return;
        }
    }
    auto timeEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = timeEnd-timeStart;
    if (timeFlag && !offlineFlag){
        outputFile << elapsed_seconds.count();
        outputFile.close();
    }
    std::printf("Elapsed time: %fs\n", std::chrono::duration<double>(timeEnd-timeStart).count());
}

void SynthUserInterface::commandMidiReaderRewind(){
    if (inputTokenCount > 1){
        short inputID;
        for (uint i = 1; i < inputTokenCount; i++){
            if (numberFromToken(i, inputID)){
                error = true;
                return;
            }
            if (audioPipeline->isMidiReader(inputID) == false){
                std::printf("MIDI INPUT(%d) does not exist\n", inputID);
                continue;
            }
            if (audioPipeline->rewindMidiReader(inputID)){
                std::printf("Something went wrong!\n");
                error = true;
                return;
            }
            std::printf("MIDI INPUT(%d) rewinded\n", inputID);
        }
    } else {
        audioPipeline->rewindMidiReaders();
        std::printf("All MIDI readers rewinded\n");
    }
}

void SynthUserInterface::commandMidiReaderList(){
    printf("MIDI readers:\n");
    audioPipeline->printMidiReaders();
}


void SynthUserInterface::commandRecordStatistics(){
    if (inputTokenCount < 2){
        std::printf("Usage: statRecord <update time interval> <file path>\n");
        error = true;
        return;
    }
    float timeInterval;
    if (numberFromToken(1, timeInterval)){
        error = true;
        return;
    }
    if (audioPipeline->recordStatistics(concatenateTokens(2), timeInterval)){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Statistics will be saved under %s\n", concatenateTokens(2).c_str());
}

void SynthUserInterface::commandStopRecordStatistics(){
    if (audioPipeline->stopRecordingStatistics()){
        std::printf("Something went wrong!\n");
        error = true;
        return;
    }
    std::printf("Statistics recording stopped\n");
}

