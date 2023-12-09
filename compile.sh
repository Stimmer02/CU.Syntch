#!/bin/bash

cd src || exit

COMPILATION_COMMAND="g++ main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer.cpp AudioPipelineSubstitute.cpp Synthesizer/Generator_Sine.cpp UserInput/keyboardTransferBuffer.cpp AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8.cpp Pipeline/BufferConverter/BufferConverter_Mono16.cpp Pipeline/BufferConverter/BufferConverter_Mono24.cpp Pipeline/BufferConverter/BufferConverter_Mono32.cpp Pipeline/BufferConverter/BufferConverter_Stereo8.cpp Pipeline/BufferConverter/BufferConverter_Stereo16.cpp Pipeline/BufferConverter/BufferConverter_Stereo24.cpp Pipeline/BufferConverter/BufferConverter_Stereo32.cpp Pipeline/Statistics/PipelineStatisticsService.cpp SynthUserInterface.cpp AudioOutput/AudioRecorder.cpp Synthesizer/Generator_Square.cpp Synthesizer/Generator_Sawtooth.cpp Synthesizer/DynamicsController.cpp UserInput/MIDI/MidiFileReader.cpp UserInput/MIDI/MidiMessageInterpreter.cpp UserInput/KeyboardRecorder_DevSnd.cpp Synthesizer/AGenerator.cpp -lpulse-simple -lpulse -ggdb -Wall"

eval "$($COMPILATION_COMMAND)"

if [ "$?" = 0 ]; then
    cd ..
    mv ./src/a.out ./CPU-synth.out

    if [ "$1" = "-r" ]; then
        echo "running: ./CPU-synth $2"
        ./CPU-synth.out "$2"
    fi
fi


