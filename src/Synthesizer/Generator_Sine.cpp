#include "Generator_Sine.h"
// #include <iostream>
void synthesizer::Generator_Sine::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings){
    uint i = 0;
    float ampitude = settings.maxValue * settings.volume;
    for (; i < settings.sampleSize; i++){
        if (keyState[i]){
            noteBuffer.lastAttack = float(settings.attack.duration-noteBuffer.samplesAfterPress)/settings.attack.duration;
            noteBuffer.samplesAfterPress += noteBuffer.lastAttack > 0;
            noteBuffer.samplesAfterRelease = 0;
            noteBuffer.buffer[i] = sin((i+noteBuffer.phaze)*noteBuffer.multiplier) * ampitude * (1-noteBuffer.lastAttack);
        } else if (noteBuffer.samplesAfterRelease < settings.release.duration) {
            float release = float(settings.release.duration-noteBuffer.samplesAfterRelease)/settings.release.duration;
            noteBuffer.buffer[i] = sin((i+noteBuffer.phaze)*noteBuffer.multiplier) * ampitude * (1-noteBuffer.lastAttack) * release;
            noteBuffer.samplesAfterPress = 0;
            noteBuffer.samplesAfterRelease++;
        } else {
            noteBuffer.phaze = 0;
            noteBuffer.buffer[i] = 0;
            noteBuffer.samplesAfterPress = 0;
        }
    }
    noteBuffer.phaze += i + 1;
}
