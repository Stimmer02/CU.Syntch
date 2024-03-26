#ifndef MIDISETTINGS_H
#define MIDISETTINGS_H

// #include <cmath>
#include <cstdint>
// #include <cstdio>

typedef unsigned long int ulong;
typedef unsigned int uint;
typedef unsigned char uchar;

namespace MIDI{
    struct midiSettings {
        uint tempo = 500000;//microseconds per beat
        double ticksPerSample;
        double tickValue;//microseconds per tick

        float calculateBPM(){
            return 60000000.0/tempo;
        }

        void calculateTickValue(uint16_t timeDivision, uint sampleRate, uint sampleSize){
            if ((timeDivision & 0x8000) == 0){
                tickValue = double(tempo)/(timeDivision & 0x7FFF);
                // printf("1 tickValue: %lf\n", tickValue);
            } else {
                uint FPS = (timeDivision & 0x7FFF) >> 8;
                uint ticksPerFrame = timeDivision & 0xFF;
                tickValue = 1000000.0 / (FPS * ticksPerFrame);
                // printf("2 tickValue: %lf\n", tickValue);
            }
            ticksPerSample = double(sampleSize)/sampleRate*1000000/tickValue;
            // printf("TPSa: %f, TV: %f\n", ticksPerSample, tickValue);
        }
    };
}

#endif
