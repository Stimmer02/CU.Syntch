#ifndef MIDIEVENT_H
#define MIDIEVENT_H

#include <cstdint>


typedef unsigned long int ulong;
typedef unsigned int uint;
typedef unsigned char uchar;


namespace MIDI{
    enum event_type{
        MIDI,
        SYSEX,
        META,
        ESC,
        NONE,
    };

    struct midiEvent{
        uint32_t deltaTime;
        uchar message[3];
        char* longerMessage;
        uint32_t messageLength;
        event_type type;
        uchar channel;

        midiEvent(){
            lMessageMaxLength = 64;
            longerMessage = new char[lMessageMaxLength];
            init();
        }

        void init(){
            deltaTime = 0;
            messageLength = 0;
            type = NONE;
            channel = 0;
        }

        ~midiEvent(){
            delete[] longerMessage;
        }

        void setMessageLength(uint32_t size){
            if (size > lMessageMaxLength){
                lMessageMaxLength = size;
                delete[] longerMessage;
                longerMessage = new char[lMessageMaxLength];
            }
            messageLength = size;
        }
    private:
        uint32_t lMessageMaxLength;
    };
}

#endif
