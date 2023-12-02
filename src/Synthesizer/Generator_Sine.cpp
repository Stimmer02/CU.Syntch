#include "Generator_Sine.h"

void synthesizer::Generator_Sine::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile){
    uint i = 0;
    for (; i < settings.sampleSize; i++){
        if (keyState[i]){
            noteBuffer.phaze++;
            noteBuffer.pressSamplessPassed++;
            noteBuffer.pressSamplessPassedCopy = noteBuffer.pressSamplessPassed;
            noteBuffer.releaseSamplesPassed = 0;
            if (settings.dynamicsDuration > noteBuffer.pressSamplessPassed){
                noteBuffer.buffer[i] = sin(noteBuffer.phaze*noteBuffer.multiplier) * settings.volume * dynamicsProfile[noteBuffer.pressSamplessPassed];
            } else {
                noteBuffer.buffer[i] = sin(noteBuffer.phaze*noteBuffer.multiplier) * settings.volume * settings.fadeTo;
            }
        } else if (noteBuffer.releaseSamplesPassed < settings.release.duration){
            noteBuffer.phaze++;
            noteBuffer.releaseSamplesPassed++;

            double dynamicsMultiplier;

            if (settings.dynamicsDuration > noteBuffer.pressSamplessPassedCopy){
                dynamicsMultiplier = dynamicsProfile[noteBuffer.pressSamplessPassedCopy] * releaseProfile[noteBuffer.releaseSamplesPassed];
            } else {
                dynamicsMultiplier = settings.fadeTo * releaseProfile[noteBuffer.releaseSamplesPassed];
            }

            noteBuffer.pressSamplessPassed = (settings.attack.duration - 1) * dynamicsMultiplier;
            noteBuffer.buffer[i] = sin(noteBuffer.phaze * noteBuffer.multiplier) * settings.volume * dynamicsMultiplier;//BUG: this line crashes if I increase release time while playing any sound

        } else {
            noteBuffer.phaze = 0;
            noteBuffer.pressSamplessPassed = 0;
            noteBuffer.buffer[i] = 0;
        }
    }
}
