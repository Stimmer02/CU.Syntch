#include "AGenerator.h"

void synthesizer::AGenerator::generate(noteBuffer& noteBuffer, const uchar* keyState, const settings& settings, const float* dynamicsProfile, const float* releaseProfile){
    uint i = 0;
    for (; i < settings.sampleSize; i++){
        if (keyState[i]){
            noteBuffer.phaze++;
            noteBuffer.pressSamplessPassed++;
            noteBuffer.pressSamplessPassedCopy = noteBuffer.pressSamplessPassed;
            noteBuffer.releaseSamplesPassed = 0;
            noteBuffer.velocity = keyState[i]/127.0;

            float dynamicsMultiplier = settings.dynamicsDuration > noteBuffer.pressSamplessPassed ? dynamicsProfile[noteBuffer.pressSamplessPassed] : settings.fadeTo;

            noteBuffer.buffer[i] = soundFunction(noteBuffer) * settings.volume * noteBuffer.velocity * dynamicsMultiplier;

        } else if (noteBuffer.releaseSamplesPassed < settings.release.duration){
            noteBuffer.phaze++;

            float dynamicsMultiplier = settings.dynamicsDuration > noteBuffer.pressSamplessPassedCopy ? dynamicsProfile[noteBuffer.pressSamplessPassedCopy] : settings.fadeTo;

            dynamicsMultiplier *= releaseProfile[noteBuffer.releaseSamplesPassed];
            noteBuffer.pressSamplessPassed = (settings.attack.duration - 1) * dynamicsMultiplier;

            noteBuffer.buffer[i] = soundFunction(noteBuffer) * settings.volume * noteBuffer.velocity * dynamicsMultiplier;//BUG: this line crashes if I increase release time while playing any sound

            noteBuffer.releaseSamplesPassed++;

        } else {
            noteBuffer.phaze = 0;
            noteBuffer.pressSamplessPassed = 0;
            noteBuffer.buffer[i] = 0;
        }
    }
}
