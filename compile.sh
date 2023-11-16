#!/bin/bash

COMPILATION_COMMAND="g++ main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer.cpp AudioPipelineSubstitute.cpp Synthesizer/Generator_Sine.cpp UserInput/keyboardTransferBuffer.cpp AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8.cpp Pipeline/BufferConverter/BufferConverter_Mono16.cpp Pipeline/BufferConverter/BufferConverter_Mono24.cpp Pipeline/BufferConverter/BufferConverter_Mono32.cpp Pipeline/Statistics/PipelineStatisticsService.cpp -lpulse-simple -lpulse -ggdb"

$($COMPILATION_COMMAND)

if [ $? = 0 ] && [ "$1" = "-r" ]; then
    ./a.out "$2"
fi
