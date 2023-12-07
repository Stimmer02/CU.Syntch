#include "Generator_Square.h"

void synthesizer::Generator_Square::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile){
    uint i = 0;
    for (; i < settings.sampleSize; i++){
        if (keyState[i]){
            noteBuffer.phaze++;
            noteBuffer.pressSamplessPassed++;
            noteBuffer.pressSamplessPassedCopy = noteBuffer.pressSamplessPassed;
            noteBuffer.releaseSamplesPassed = 0;
            noteBuffer.velocity = keyState[i]/127.0;
            if (settings.dynamicsDuration > noteBuffer.pressSamplessPassed){
                noteBuffer.buffer[i] = ((int((noteBuffer.phaze)/noteBuffer.multiplier) & 0x1)*2 - 1) * settings.volume * noteBuffer.velocity * dynamicsProfile[noteBuffer.pressSamplessPassed];
            } else {
                noteBuffer.buffer[i] = ((int((noteBuffer.phaze)/noteBuffer.multiplier) & 0x1)*2 - 1) * settings.volume * noteBuffer.velocity * settings.fadeTo;
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
            noteBuffer.buffer[i] = ((int((noteBuffer.phaze)/noteBuffer.multiplier) & 0x1)*2 - 1) * settings.volume * noteBuffer.velocity * dynamicsMultiplier;

        } else {
            noteBuffer.phaze = 0;
            noteBuffer.pressSamplessPassed = 0;
            noteBuffer.buffer[i] = 0;
        }
    }
}
