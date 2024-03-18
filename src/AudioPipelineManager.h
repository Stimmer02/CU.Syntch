#ifndef AUDIOPIPELINEMANAGER_H
#define AUDIOPIPELINEMANAGER_H

#include "Pipeline/ComponentManager.h"
#include "Pipeline/Components/componentSettings.h"
#include "Pipeline/IDManager.h"
#include "Pipeline/Input.h"
#include "Pipeline/Output.h"
#include "Pipeline/Statistics/PipelineStatisticsService.h"
#include "Pipeline/Statistics/pipelineStatistics.h"
#include "Pipeline/ExecutionQueue.h"
#include "Pipeline/audioBufferQueue.h"
#include "Pipeline/pipelineAudioBuffer.h"
#include "Synthesizer.h"
#include "UserInput/MIDI/MidiFileReader.h"
#include "enumConversion.h"



namespace pipeline{
    class AudioPipelineManager{
    public:
        AudioPipelineManager(audioFormatInfo audioInfo, ushort keyCount);
        ~AudioPipelineManager();

        char start();
        void stop();
        bool isRuning();

        const statistics::pipelineStatistics* getStatistics();
        const audioFormatInfo* getAudioInfo();

        char recordUntilStreamEmpty(MIDI::MidiFileReader& midi, short synthID, std::string filename = "");//TODO: rethink this function
        bool IDValid(pipeline::ID_type type, short ID);
        void reorganizeIDs();

        void emptyQueueBuffer(ID_type IDType, short ID);



        //OUTPUT CONTROL
        char startRecording();
        char startRecording(std::string outPath);
        char stopRecording();
        bool isRecording();


        //INPUT CONTROL
        char pauseInput();
        char reausumeInput();
        void clearInputBuffers();

        short addInput(AKeyboardRecorder*& input);
        char removeInput(short inputID);
        short getInputCount();

        short addSynthesizer();
        char removeSynthesizer(short synthID);
        short getSynthesizerCount();

        char connectInputToSynth(short inputID, short synthID);
        char disconnectSynth(short synthID);

        const synthesizer::settings* getSynthSettings(const ushort& synthID);
        synthesizer::generator_type getSynthType(const ushort& synthID);
        float getSynthSetting(const ushort& synthID, synthesizer::settings_name settingName);
        void setSynthSetting(const ushort& synthID, const synthesizer::settings_name& settingsName, const float& value);
        void setSynthSetting(const ushort& synthID, const synthesizer::generator_type& type);

        char saveSynthConfig(std::string path, ushort synthID);
        char loadSynthConfig(std::string path, ushort synthID);

        char printSynthInfo(short synthID);

        //COMPONENT CONTROL

        char setOutputBuffer(short ID, ID_type IDType);

        short addComponent(component_type type);
        short addComponent(advanced_component_type type);

        char removeComponent(short componentID);
        char removeSimpleComponent(short componentID);
        char removeAdvancedComponent(short componentID);

        char disconnectCommponent(short componentID);
        void disconnectSimpleCommponent(short componentID);
        void disconnectAdvancedCommponentFromAll(short componentID);
        void disconnectAdvancedCommponent(short componentID, ID_type parentType, short parentID);
        void disconnectAdvancedCommponent(short componentID, uint index);
        char tryDisconnectAdvancedCommponent(short componentID, uint index);

        short getComponentCout();
        char connectComponent(short componentID, ID_type parentType, short parentID);
        char setAdvancedComponentInput(short componentID, short inputIndex, ID_type IDType, short connectToID);
        char getComponentConnection(short componentID, ID_type& parentType, short& parentID);
        char setComponentSetting(short componentID, uint settingIndex, float value);
        const componentSettings* getComopnentSettings(short componentID);

        bool isAdvancedComponent(short ID);

        char printAdvancedComponentInfo(short componentID);


    private:
        int findAudioQueueBufferIndex(ID_type IDType, short ID);
        audioBufferQueue* findAudioQueueBuffer(ID_type IDType, short ID);

        void pipelineThreadFunction();

        const audioFormatInfo audioInfo;
        const ushort keyCount;

        Input input;//synths, inputs
        Output output;//audio output, recording

        ComponentManager component;//component collection, processing
        std::vector<audioBufferQueue*> componentQueues;//audio buffers, processing order of any queue
        audioBufferQueue* outputBuffer;//the last componentQueue to be executed
        ExecutionQueue executionQueue;//processing order of componentQueues

        statistics::PipelineStatisticsService* statisticsService;

        bool running;
        std::thread* pipelineThread;
    };
}

#endif
