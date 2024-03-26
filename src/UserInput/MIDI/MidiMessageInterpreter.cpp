#include "MidiMessageInterpreter.h"

using namespace MIDI;

char MidiMessageInterpreter::getVariableLengthValue(std::ifstream* stream, uint32_t& out){
    uchar byte;
    out = 0;
    do {
        stream->get((char&)(byte));
        if (stream->eof()){
            return 1;
        }
        out <<= 7;
        out |= byte & 0x7F;
    } while (byte & 0x80);
    return 0;
}

char MidiMessageInterpreter::getFileEvent(std::ifstream* stream, midiEvent& event){
    if (getVariableLengthValue(stream, event.deltaTime)){
        return 1;
    }
    return getEvent(stream, event);
}

char MidiMessageInterpreter::getEvent(std::ifstream* stream, midiEvent& event){

    stream->get((char&)(event.message[0]));

    if (event.message[0] >= 0x80 && event.message[0] < 0xF0){
        event.type = MIDI;
        event.channel = event.message[0] & 0x0F;
        stream->get((char&)(event.message[1]));
        if (event.message[1] < 0xC0 || event.message[1] >= 0xE0){
            stream->get((char&)(event.message[2]));
        }
    } else if (event.message[0] == 0xF0){
        event.type = SYSEX;
        ignoreSysEx(stream);
    } else if (event.message[0] == 0xF7){
        event.type = ESC;
        uint32_t messageLength;
        getVariableLengthValue(stream, messageLength);
        if (getLongerMessage(stream, event, messageLength)){
            return 1;
        }
    } else if (event.message[0] == 0xFF){
        event.type = META;
        stream->get((char&)(event.message[1]));
        uint32_t messageLength;
        getVariableLengthValue(stream, messageLength);
        if (getLongerMessage(stream, event, messageLength)){
            return 1;
        }
    } else {
        //outside of the MIDI standard range: 0x80 - 0xFF
        event.type = NONE;
    }

    if (stream->eof()){
        return 1;
    }
    return 0;
}

void MidiMessageInterpreter::executeMidiEvent(const midiEvent& event, uchar* buffer[127], uint timePlacement){
    switch (event.message[0] & 0xF0){
        case 0x80:
            buffer[event.message[1]][timePlacement] = 255;
            break;

        case 0x90:
            buffer[event.message[1]][timePlacement] = event.message[2] == 0 ? 255 : event.message[2];
            break;

        case 0xA0:
            buffer[event.message[1]][timePlacement] = event.message[2] == 0 ? 255 : event.message[2];
            break;

        case 0xB0:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeMidiEvent: Controller unimplemented\n");
            break;

        case 0xC0:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeMidiEvent: Program change unimplemented\n");
            break;

        case 0xD0:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeMidiEvent: Channel pressure unimplemented\n");
            break;

        case 0xE0:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeMidiEvent: Pitch / modulation wheel unimplemented\n");
            break;

        case 0x00:
            //this happens before the first event is even read
            break;

        default:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeMidiEvent: unrecognised MIDI event: 0x%02x\n", event.message[0]);
            break;
    }
}

char MidiMessageInterpreter::executeEvent(const midiEvent& event, uchar* buffer[127], midiSettings& settings, uint timePlacement, const uint& sampleSize, const uint& sampleRate, const midiCheaderChunk& info){
    switch (event.type){
        case MIDI:
            executeMidiEvent(event, buffer, timePlacement);
            break;

        case SYSEX:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeEvent: unrecognised SYSEX event: 0x%02x\n", event.message[0]);
            break;

        case META:
            switch (event.message[1]){
                case 0x00: //Sequence Number
                    printf("MIDI INFO [sequence number]: %s\n", event.longerMessage);
                    break;

                case 0x01: //Text
                    printf("MIDI INFO [text]: %s\n", event.longerMessage);
                    break;

                case 0x02: //Copyright
                    printf("MIDI INFO [copyright]: %s\n", event.longerMessage);
                    break;

                case 0x03: //Sequence / Track Name
                    printf("MIDI INFO [sequence / track name]: %s\n", event.longerMessage);
                    break;

                case 0x04: //Instrument Name
                    printf("MIDI INFO [instrument name]: %s\n", event.longerMessage);
                    break;

                case 0x05: //Lyric
                    printf("MIDI INFO [lyric]: %s\n", event.longerMessage);
                    break;

                case 0x06: //Marker
                    printf("MIDI INFO [marker]: %s\n", event.longerMessage);
                    break;

                case 0x07: //Cue Point
                    printf("MIDI INFO [cue point]: %s\n", event.longerMessage);
                    break;

                case 0x08: //Program Name
                    printf("MIDI INFO [program name]: %s\n", event.longerMessage);
                    break;

                case 0x09: //Device Name
                    printf("MIDI INFO [device name]: %s\n", event.longerMessage);
                    break;

                // case 0x20: //MIDI Channel Prefix
                    //select channel
                    // break;

                // case 0x21: //MIDI Port
                    //select port
                    // break;

                case 0x2F: //End of Track
                    return 1;

                case 0x51: //Tempo
                    settings.tempo = (event.longerMessage[0] << 16) + (event.longerMessage[1] << 8) + event.longerMessage[2];
                    settings.calculateTickValue(info.timeDivision, sampleRate, sampleSize);
                    std::printf("Tempo set to %.02f BPM\n", settings.calculateBPM());
                    break;

                // case 0x54: //SMPTE Offset
                    // break;

                // case 0x58: //Time Signature
                    // break;

                // case 0x59: //Key Signature
                    // break;

                // case 0x7F: //Sequencer Specific Event


                default:
                    std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeEvent: unrecognised META event: 0x%02x\n", event.message[1]);
                    break;
            }
            break;

        case ESC:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeEvent: unrecognised ESC event: 0x%02x\n", event.message[0]);
            break;

        case NONE:
            std::fprintf(stderr, "WARNING: MidiMessageInterpreter::executeEvent: unrecognised event: 0x%02x\n", event.message[0]);
            break;
    }

    return 0;
}

char MidiMessageInterpreter::ignoreSysEx(std::ifstream* stream){
    uchar byte;
    do {
        stream->get((char&)(byte));
        if (stream->eof()){
            return 1;
        }
    } while (byte != 0xF7);
    return 0;
}



char MidiMessageInterpreter::getLongerMessage(std::ifstream* stream, midiEvent& event, uint32_t length){
    event.setMessageLength(length+1);
    stream->read(event.longerMessage, length);
    event.longerMessage[length] = 0;
    return 0;
}
