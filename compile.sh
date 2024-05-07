#!/bin/bash

cd src || exit

g++ main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer.cpp Synthesizer/Generator_Sine.cpp UserInput/keyboardTransferBuffer.cpp AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8.cpp Pipeline/BufferConverter/BufferConverter_Mono16.cpp Pipeline/BufferConverter/BufferConverter_Mono24.cpp Pipeline/BufferConverter/BufferConverter_Mono32.cpp Pipeline/BufferConverter/BufferConverter_Stereo8.cpp Pipeline/BufferConverter/BufferConverter_Stereo16.cpp Pipeline/BufferConverter/BufferConverter_Stereo24.cpp Pipeline/BufferConverter/BufferConverter_Stereo32.cpp Pipeline/Statistics/PipelineStatisticsService.cpp SynthUserInterface.cpp AudioOutput/AudioRecorder.cpp Synthesizer/Generator_Square.cpp Synthesizer/Generator_Sawtooth.cpp Synthesizer/DynamicsController.cpp UserInput/MIDI/MidiFileReader.cpp UserInput/MIDI/MidiMessageInterpreter.cpp UserInput/KeyboardRecorder_DevSnd.cpp Synthesizer/AGenerator.cpp Synthesizer/Generator_Noise1.cpp Synthesizer/Generator_Triangle.cpp Pipeline/Input.cpp Pipeline/Output.cpp AudioPipelineManager.cpp Pipeline/ComponentManager.cpp Pipeline/audioBufferQueue.cpp UserInput/TerminalHistory.cpp enumConversion.cpp Pipeline/ExecutionQueue.cpp UserInput/ScriptReader.cpp Pipeline/Components/AComponent.cpp Pipeline/Components/Component_Volume.cpp Pipeline/Components/AAdvancedComponent.cpp Pipeline/Components/AdvancedComponent_Sum2.cpp Pipeline/Components/Component_Pan.cpp Pipeline/Components/Component_Echo.cpp Pipeline/Components/Component_Distortion.cpp Pipeline/Components/Component_Compressor.cpp Pipeline/Components/Component_Destroy.cpp Pipeline/Components/AdvancedComponent_Copy.cpp UserInput/KeyboardDoubleBuffer_Empty.cpp UserInput/MIDI/KeyboardDoubleBuffer_MidiFile.cpp UserInput/MIDI/KeyboardRecorder_MidiFile.cpp UserInput/MIDI/MidiReaderManager.cpp Pipeline/Components/AdvancedComponent_Sum7.cpp Pipeline/AudioSpectrumVisualizer.cpp -lpulse-simple -lpulse -Wall -mavx2 -lfftw3 #-DNO_AVX2

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
