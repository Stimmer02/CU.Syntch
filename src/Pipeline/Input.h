#ifndef PIPELINEINPUT_H
#define PIPELINEINPUT_H

#include "../Synthesizer.h"
#include "KeyboardManager.h"
#include "KeyboardManager.cpp"
#include "pipelineAudioBuffer.h"

namespace pipeline{
    /*
    ID explanation:
        1. there are two pools of ID's:
            -input (midiInput, keyboardsState)
            -synthesizers (synths)
            -components (not here)
        2. ID is an index in (input/synth)IDMap
        3. IDMap contains indexes of objects in their storing array (midiInput, etc.)
        4. ID can only have value of not negative range
        5. negative ID is indication of an error:
            -1: out of ID's?
            -2: object initialization error
        6. after object is deleted it's ID becomes empty (IDMap will store -1) and any reference to it will raturn an error
        7. Every synthesizer has its own audioBufferQueue - creates the source signall


    */

    class Input{
    public:
        Input();
        ~Input();

        char init(audioFormatInfo audioInfo, ushort keyCount);

        char startAllInputs();
        char stopAllInputs();
        void clearBuffers();

        short addInput(AKeyboardRecorder* input);
        char removeInput(short ID);
        short getInputCount();
        long getActivationTimestamp();

        void swapActiveBuffers();
        void cycleBuffers();

        short addSynthesizer(pipelineAudioBuffer* buffer);
        char removeSynthesizer(short ID);
        short getSynthesizerCount();
        void setSynthetiserSetting(short ID, synthesizer::settings_name settingsName, float value);
        const synthesizer::settings* getSynthetiserSettins(short ID);

        char connectInputToSynth(short inputID, short synthID);
        char disconnectSynth(short synthID);

        void generateSamples();
        void generateSampleWith(short synthID, pipelineAudioBuffer* buffer, keyboardTransferBuffer* keyboardState);

        char saveSynthConfig(std::string path, short ID);
        char loadSynthConfig(std::string path, short ID);

        void reorganizeIDs();//gives every object an ID in the same order they are being stored
        bool synthIDValid(short ID);
        bool inputIDValid(short ID);
        void removeAll();

    private:
        struct synthWithConnection{
            synthWithConnection(pipelineAudioBuffer* buffer, audioFormatInfo audioInfo, ushort keyCount):synth(audioInfo, keyCount){
                midiInputID = -2;
                this->buffer = buffer;
            };
            synthesizer::Synthesizer synth;
            short midiInputID;
            pipelineAudioBuffer* buffer;
        };

        void cleanup();

        audioFormatInfo audioInfo;
        ushort keyCount;

        KeyboardManager<short> midiInput;
        IDManager<synthWithConnection, short> synths;

        bool running;
    };
}

#endif
