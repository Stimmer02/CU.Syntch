#include "Generator_Sawtooth.h"

void synthesizer::Generator_Sawtooth::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile){
    uint i = 0;
    for (; i < settings.sampleSize; i++){//TODO: seems as it is little bit out of tune
        if (keyState[i]){
            noteBuffer.phaze++;
            noteBuffer.pressSamplessPassed++;
            noteBuffer.pressSamplessPassedCopy = noteBuffer.pressSamplessPassed;
            noteBuffer.releaseSamplesPassed = 0;
            noteBuffer.velocity = keyState[i]/127.0;
            if (settings.dynamicsDuration > noteBuffer.pressSamplessPassed){
                noteBuffer.buffer[i] = (((noteBuffer.phaze)%uint(noteBuffer.multiplier))/noteBuffer.multiplier*2 - 1) * settings.volume * noteBuffer.velocity * dynamicsProfile[noteBuffer.pressSamplessPassed];
            } else {
                noteBuffer.buffer[i] = (((noteBuffer.phaze)%uint(noteBuffer.multiplier))/noteBuffer.multiplier*2 - 1) * settings.volume * noteBuffer.velocity * settings.fadeTo;
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
            noteBuffer.buffer[i] = (((noteBuffer.phaze)%uint(noteBuffer.multiplier))/noteBuffer.multiplier*2 - 1) * settings.volume * noteBuffer.velocity * dynamicsMultiplier;

        } else {
            noteBuffer.phaze = 0;
            noteBuffer.pressSamplessPassed = 0;
            noteBuffer.buffer[i] = 0;
        }
    }
}
