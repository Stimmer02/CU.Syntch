#ifndef _NOTEBUFFER_H
#define _NOTEBUFFER_H

typedef unsigned int uint;
typedef unsigned char uchar;

namespace synthesizer{
    struct noteBuffer{
        double* buffer;

        double velocity;

        uint phaze;
        uint pressSamplessPassed;
        uint pressSamplessPassedCopy;
        uint releaseSamplesPassed;

        double stereoFactorL;
        double stereoFactorR;

        float frequency;
        float multiplier;


        noteBuffer(){
            buffer = nullptr;
            phaze = 0;
            pressSamplessPassed = 0;
            pressSamplessPassedCopy = 0;
            releaseSamplesPassed = 0;
        }
        noteBuffer(const uint& bufferSize){
            buffer = new double[bufferSize];
            // state = new uchar[bufferSize];
        }
        void init(const uint& bufferSize){
            if (buffer != nullptr){
                delete[] buffer;
                // delete[] state;
            }
            buffer = new double[bufferSize];
            // state = new uchar[bufferSize];
        }
        ~noteBuffer(){
            if (buffer != nullptr){
                delete[] buffer;
                // delete[] state;
            }
        }
    };
}

#endif
