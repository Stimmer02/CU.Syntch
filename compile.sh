#!/bin/bash

cd src || exit

COMPILATION_COMMAND="g++ main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer.cpp AudioPipelineSubstitute.cpp Synthesizer/Generator_Sine.cpp UserInput/keyboardTransferBuffer.cpp AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8.cpp Pipeline/BufferConverter/BufferConverter_Mono16.cpp Pipeline/BufferConverter/BufferConverter_Mono24.cpp Pipeline/BufferConverter/BufferConverter_Mono32.cpp Pipeline/Statistics/PipelineStatisticsService.cpp SynthUserInterface.cpp -lpulse-simple -lpulse -ggdb"

eval "$($COMPILATION_COMMAND)"

if [ "$?" = 0 ]; then
    cd ..
    mv ./src/a.out ./CPU-synth.out

    if [ "$1" = "-r" ]; then
        echo "running: ./CPU-synth $2"
        ./CPU-synth.out "$2"
    fi
fi


