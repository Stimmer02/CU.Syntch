#!/bin/bash

cd src || exit

nvcc main.cpp UserInput/KeyboardInput_DevInput.cpp UserInput/TerminalInputDiscard.cpp UserInput/KeyboardRecorder_DevInput.cpp UserInput/KeyboardDoubleBuffer.cpp Synthesizer_CUDA.cu UserInput/keyboardTransferBuffer_CUDA.cu AudioOutput/OutStream_PulseAudio.cpp AudioOutput/audioBuffer.cpp UserInput/InputMap.cpp Pipeline/BufferConverter/BufferConverter_Mono8_CUDA.cu Pipeline/BufferConverter/BufferConverter_Mono16_CUDA.cu Pipeline/BufferConverter/BufferConverter_Mono24_CUDA.cu Pipeline/BufferConverter/BufferConverter_Mono32_CUDA.cu Pipeline/BufferConverter/BufferConverter_Stereo8_CUDA.cu Pipeline/BufferConverter/BufferConverter_Stereo16_CUDA.cu Pipeline/BufferConverter/BufferConverter_Stereo24_CUDA.cu Pipeline/BufferConverter/BufferConverter_Stereo32_CUDA.cu Pipeline/Statistics/PipelineStatisticsService.cpp SynthUserInterface.cpp AudioOutput/AudioRecorder.cpp Synthesizer/DynamicsController_CUDA.cu UserInput/MIDI/MidiFileReader.cpp UserInput/MIDI/MidiMessageInterpreter.cpp UserInput/KeyboardRecorder_DevSnd.cpp Synthesizer/Generator_CUDA.cu Pipeline/Input.cpp Pipeline/Output.cpp AudioPipelineManager.cpp Pipeline/ComponentManager.cpp Pipeline/audioBufferQueue.cpp UserInput/TerminalHistory.cpp enumConversion.cpp Pipeline/ExecutionQueue.cpp UserInput/ScriptReader.cpp Pipeline/Components/AComponent_CUDA.cpp Pipeline/Components/Component_Volume_CUDA.cu Pipeline/Components/AAdvancedComponent_CUDA.cpp Pipeline/Components/AdvancedComponent_Sum2_CUDA.cu Pipeline/Components/Component_Pan_CUDA.cu Pipeline/Components/Component_Echo_CUDA.cu Pipeline/Components/Component_Distortion_CUDA.cu Pipeline/Components/Component_SimpleCompressor_CUDA.cu Pipeline/Components/Component_Destroy_CUDA.cu Pipeline/Components/AdvancedComponent_Copy_CUDA.cu UserInput/KeyboardDoubleBuffer_Empty.cpp UserInput/MIDI/KeyboardDoubleBuffer_MidiFile.cpp UserInput/MIDI/KeyboardRecorder_MidiFile.cpp UserInput/MIDI/MidiReaderManager.cpp Pipeline/Components/AdvancedComponent_Sum7_CUDA.cu Pipeline/AudioSpectrumVisualizer.cpp Synthesizer/NoteBufferHandler_CUDA.cpp -lpulse-simple -lpulse -lfftw3 -g -G #-Wall

RETURN_CODE=$?

if [ "$RETURN_CODE" -eq 0 ]; then
    cd ..
    mv ./src/a.out ./GPU-synth.out

    if [ "$1" = "-r" ]; then
        echo "running: ./GPU-synth $2"
        ./CPU-synth.out "$2"
        RETURN_CODE=$?
    fi
fi

exit $RETURN_CODE
