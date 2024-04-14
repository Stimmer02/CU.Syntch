#ifndef SYNTHUSERINTERFACE_H
#define SYNTHUSERINTERFACE_H

#include "AudioPipelineManager.h"
#include "UserInput/TerminalInputDiscard.h"
#include "UserInput/TerminalHistory.h"
#include "UserInput/InputMap.h"
#include "UserInput/KeyboardRecorder_DevInput.h"
#include "UserInput/KeyboardRecorder_DevSnd.h"
#include "enumConversion.h"
#include "UserInput/ScriptReader.h"
#include "UserInput/IScriptReaderClient.h"
#include "UserInput/KeyboardInput_DevInput.h"


#include <linux/input-event-codes.h>
#include <iostream>
#include <map>


namespace pipeline{
    class SynthUserInterface : private IScriptReaderClient{
    public:
        SynthUserInterface(std::string terminalHistoryPath, audioFormatInfo audioInfo, ushort keyCount);
        ~SynthUserInterface();

        char setUserInput(IKeyboardInput*& userInput);

        char start();
        bool getErrorFlag() override;
        void clearErrorFlag() override;

        ScriptReader scriptReader;

    private:
        void parseInput();
        void readInput();
        void executeCommand(std::string& command) override;
        void waitUntilKeyReleased(ushort key);

        IKeyboardInput* userInput;
        AudioPipelineManager* audioPipeline;
        TerminalInputDiscard terminalDiscard;
        TerminalHistory history;

        bool running;
        uint loopDelay;
        ushort keyCount;


        bool terminalInput;
        bool error;

        std::string inputLine;
        const ushort inputTokenMax = 64;
        const char** inputTokens;
        ushort inputTokenCount;


        struct cmp_str{
            bool operator()(const char* a, const char* b) const{
                return std::strcmp(a, b) < 0;
            }
        };

        typedef void (SynthUserInterface::*methodPtr)();
        std::map<const char*, methodPtr, cmp_str>* commandMap;

        void initializeCommandMap();

        template <typename INTEGER, typename = std::enable_if_t<std::is_integral_v<INTEGER>>>
        char numberFromToken(short tokenIndex, INTEGER& out);
        char numberFromToken(short tokenIndex, float& out);
        std::string concatenateTokens(short startTokenIndex);
        void stopPipeline();

        void browseHistory();

        void commandSystem();
        void commandSetUserInput();

        void commandExit();
        void commandToggle();
        void commandHelp();
        void commandPipelineStart();
        void commandPipelineStop();
        void commandMidiRecord();
        void commandExecuteScript();
        void commandSetOutputBuffer();
        void commandClear();

        void commandVisualizerModify();
        void commandVisualizerSettings();

        void commandSynthAdd();
        void commandSynthRemove();
        void commandSynthCount();
        void commandSynthModify();
        void commandSynthSettings();
        void commandSynthSave();
        void commandSynthList();//TODO
        void commandSynthInfo();
        void commandSynthTypes();

        void commandInputAdd();
        void commandInputRemove();
        void commandInputCount();
        void commandInputList();//TODO

        void commandMidiReaderAdd();
        void commandMidiReaderSet();
        void commandMidiReaderPlay();
        void commandMidiReaderPause();
        void commandMidiReaderRecord();
        void commandMidiReaderRewind();
        void commandMidiReaderList();

        void commandSynthConnect();
        void commandSynthDisconnect();
        void commandReinitializeID();

        void commandComponentAdd();
        void commandComponentRemove();
        void commandComponentCount();
        void commandComponentConnect();
        void commandComponentDisconnect();
        void commandComponentGetConnection();
        void commandComponentModify();
        void commandComponentSettings();
        void commandComponentTypes();

        void commandAdvComponentConnect();
        void commandAdvComponentDisconnect();
        void commandAdvComponentInfo();

        void commandPrintQueues();//TODO
    };
}
#endif
