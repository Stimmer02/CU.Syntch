#ifndef _NOTEBUFFER_H
#define _NOTEBUFFER_H

typedef unsigned int uint;
typedef unsigned char uchar;

namespace synthesizer{
    struct noteBuffer{
        double* buffer;
        // uchar* state;
        float lastAttack;
        uchar velocity;
        uint phaze;

        float frequency;
        float multiplier;

        uint samplesAfterPress;
        uint samplesAfterRelease;

        noteBuffer(){
            buffer = nullptr;
            lastAttack = 1;
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
