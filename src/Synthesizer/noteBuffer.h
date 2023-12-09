#ifndef NOTEBUFFER_H
#define NOTEBUFFER_H

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
            stereoFactorL = 1;
            stereoFactorR = 0;
            frequency = 0;
            multiplier = 0;
        }
        noteBuffer(const uint& bufferSize){
            buffer = new double[bufferSize];
            phaze = 0;
            pressSamplessPassed = 0;
            pressSamplessPassedCopy = 0;
            releaseSamplesPassed = 0;
            stereoFactorL = 1;
            stereoFactorR = 0;
            frequency = 0;
            multiplier = 0;
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
