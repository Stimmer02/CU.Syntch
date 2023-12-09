#include "AGenerator.h"

void synthesizer::AGenerator::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const double* dynamicsProfile, const double* releaseProfile){
    uint i = 0;
    for (; i < settings.sampleSize; i++){
        if (keyState[i]){
            noteBuffer.phaze++;
            noteBuffer.pressSamplessPassed++;
            noteBuffer.pressSamplessPassedCopy = noteBuffer.pressSamplessPassed;
            noteBuffer.releaseSamplesPassed = 0;
            noteBuffer.velocity = keyState[i]/127.0;

            double dynamicsMultiplier = settings.dynamicsDuration > noteBuffer.pressSamplessPassed ? dynamicsProfile[noteBuffer.pressSamplessPassed] : settings.fadeTo;

            noteBuffer.buffer[i] = soundFunction(noteBuffer) * settings.volume * noteBuffer.velocity * dynamicsMultiplier;

        } else if (noteBuffer.releaseSamplesPassed < settings.release.duration){
            noteBuffer.phaze++;
            noteBuffer.releaseSamplesPassed++;

            double dynamicsMultiplier = settings.dynamicsDuration > noteBuffer.pressSamplessPassedCopy ? dynamicsProfile[noteBuffer.pressSamplessPassedCopy] : settings.fadeTo;


            noteBuffer.pressSamplessPassed = (settings.attack.duration - 1) * dynamicsMultiplier;
            noteBuffer.buffer[i] = soundFunction(noteBuffer) * settings.volume * noteBuffer.velocity * releaseProfile[noteBuffer.releaseSamplesPassed] * dynamicsMultiplier;//BUG: this line crashes if I increase release time while playing any sound

        } else {
            noteBuffer.phaze = 0;
            noteBuffer.pressSamplessPassed = 0;
            noteBuffer.buffer[i] = 0;
        }
    }
}
