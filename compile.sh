#!/bin/bash

cd src || exit

g++ main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer.cpp AudioPipelineSubstitute.cpp Synthesizer/Generator_Sine.cpp UserInput/keyboardTransferBuffer.cpp AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8.cpp Pipeline/BufferConverter/BufferConverter_Mono16.cpp Pipeline/BufferConverter/BufferConverter_Mono24.cpp Pipeline/BufferConverter/BufferConverter_Mono32.cpp Pipeline/BufferConverter/BufferConverter_Stereo8.cpp Pipeline/BufferConverter/BufferConverter_Stereo16.cpp Pipeline/BufferConverter/BufferConverter_Stereo24.cpp Pipeline/BufferConverter/BufferConverter_Stereo32.cpp Pipeline/Statistics/PipelineStatisticsService.cpp SynthUserInterface.cpp AudioOutput/AudioRecorder.cpp Synthesizer/Generator_Square.cpp Synthesizer/Generator_Sawtooth.cpp Synthesizer/DynamicsController.cpp UserInput/MIDI/MidiFileReader.cpp UserInput/MIDI/MidiMessageInterpreter.cpp UserInput/KeyboardRecorder_DevSnd.cpp Synthesizer/AGenerator.cpp Synthesizer/Generator_Noise1.cpp Synthesizer/Generator_Triangle.cpp Pipeline/Input.cpp Pipeline/Output.cpp Pipeline/IDManager.cpp Pipeline/KeyboardManager.cpp AudioPipelineManager.cpp Pipeline/ComponentManager.cpp Pipeline/AudioBufferQueue.cpp UserInput/TerminalHistory.cpp -lpulse-simple -lpulse -ggdb -Wall -mavx2 #-DNO_AVX2

RETURN_CODE=$?

if [ "$RETURN_CODE" -eq 0 ]; then
    cd ..
    mv ./src/a.out ./CPU-synth.out

    if [ "$1" = "-r" ]; then
        echo "running: ./CPU-synth $2"
        ./CPU-synth.out "$2"
        RETURN_CODE=$?
    fi
fi

exit $RETURN_CODE
