#include "MidiFileReader.h"


using namespace MIDI;

MidiFileReader::MidiFileReader(std::string path, uint sampleSize, uint sampleRate) : sampleSize(sampleSize), sampleRate(sampleRate){
    for (uint i = 0; i < 127; i++){
        tempNoteBuffer[i] = new uchar[sampleSize];
    }

    file = new std::ifstream(path, std::ios::in | std::ios::binary);
    if (file->is_open() == false){
        fileReady = false;
        std::fprintf(stderr, "ERR: MidiFileReader::MidiFileReader: COULD NOT OPEN FILE %s\n", path.c_str());
        return;
    }

    fileReady = true;
    chunks = nullptr;
    if (parseFile() != 0){
        close();
    }

    observer = nullptr;
}



MidiFileReader::~MidiFileReader(){
    for (uint i = 0; i < 127; i++){
        delete[] tempNoteBuffer[i];
    }
    close();
}


char MidiFileReader::close(){
    if (fileReady == false){
        return 1;
    }
    file->close();
    if (chunks != nullptr){
        delete[] chunks;
        delete[] lastEvent;
        delete[] chunkTime;
        delete[] lastEventTime;
        delete[] endOfChunk;
    }
    fileReady = false;

    return 0;
}

char MidiFileReader::rewindChunk(ushort chunkNumber){
    if (fileReady == false){
        return 1;
    }
    chunks[chunkNumber].lastPosition = chunks[chunkNumber].dataPosition;
    endOfChunk[chunkNumber] = false;
    chunkTime[chunkNumber] = 0;
    lastEventTime[chunkNumber] = 0;
    lastEvent[chunkNumber].init();
    return 0;
}

char MidiFileReader::rewindFile(){
    if (fileReady == false){
        return 1;
    }

    for (uint i = 0; i < info.trackCount; i++){
        chunks[i].lastPosition = chunks[i].dataPosition;
        endOfChunk[i] = false;
        chunkTime[i] = 0;
        lastEventTime[i] = 0;
        lastEvent[i].init();
    }

    return 0;
}

bool MidiFileReader::isFileReady(){
    return fileReady;
}

bool MidiFileReader::eofChunk(ushort chunkNumber){
    return endOfChunk[chunkNumber];
}

void MidiFileReader::fillBuffer(ushort chunkNumber){
    static bool emptyBuffer = false;

    if (emptyBuffer == false){
        for (uint i = 0; i < 127; i++){
            std::memset(tempNoteBuffer[i], 0, sampleSize);
        }
        emptyBuffer = true;
    }

    chunkTime[chunkNumber] += settings.ticksPerSample;
    file->seekg(chunks[chunkNumber].lastPosition);

    while (lastEventTime[chunkNumber] < chunkTime[chunkNumber]){
        if (interpreter.executeEvent(lastEvent[chunkNumber], tempNoteBuffer, settings, eventTimePlacement(chunkNumber), sampleSize, sampleRate, info)){
            endOfChunk[chunkNumber] = true;
            return;
        }
        emptyBuffer = false;
        if (interpreter.getFileEvent(file, lastEvent[chunkNumber])){
            std::fprintf(stderr, "ERR: MidiFileReader::fillBuffer: UNEXPECTED END OF FILE REACHED - CLOSING FILE\n");
            close();
            return;
        }
        lastEventTime[chunkNumber] += lastEvent[chunkNumber].deltaTime;
    }
    chunks[chunkNumber].lastPosition = file->tellg();
}

void MidiFileReader::fillBuffer(keyboardTransferBuffer* buffer, ushort chunkNumber){
    fillBuffer(chunkNumber);
    buffer->convertBuffer(tempNoteBuffer);
}

uint MidiFileReader::eventTimePlacement(ushort chunkNumber){
    return sampleSize * ((lastEventTime[chunkNumber] + settings.ticksPerSample - chunkTime[chunkNumber]) / settings.ticksPerSample);
}

void MidiFileReader::readReverse(void* out, uint byteCount){
    static uint allocated = 0;
    static std::unique_ptr<char[]> temp;

    if (byteCount > allocated){
        allocated = byteCount;
        temp.reset(new char[allocated]);
    }

    file->read(temp.get(), byteCount);
    for (uint i = 0, j = byteCount-1; i < byteCount; i++, j--){
        ((char*)out)[i] = temp.get()[j];
    }
}

void MidiFileReader::readReverse(uint16_t& out){
    file->read(reinterpret_cast<char*>(&out), sizeof(uint16_t));
    out = (out >> 8) | (out << 8);
}

void MidiFileReader::readReverse(uint32_t& out){
    file->read(reinterpret_cast<char*>(&out), sizeof(uint32_t));
    out = (out >> 24) | (out << 24) | ((out & 0x00FF0000) >> 8) | ((out & 0x0000FF00) << 8);
}

void MidiFileReader::readReverse(uint64_t& out){
    file->read(reinterpret_cast<char*>(&out), sizeof(uint64_t));
    out = (out >> 56) | (out << 56) | ((out & 0x00FF000000000000) >> 40) | ((out & 0x000000000000FF00) << 40) | ((out & 0x0000FF0000000000) >> 24) | ((out & 0x0000000000FF0000) << 24) | ((out & 0x000000FF00000000) >> 8) | ((out & 0x00000000FF000000) << 8);
}


char MidiFileReader::parseFile(){
    file->read(info.ID, 4);
    if (std::strcmp(info.ID, "MThd") != 0){
        std::fprintf(stderr, "ERR MidiFileReader::parseFile: SPECIFIED FILE DOES NOT HAVE PROPER MIDI HEADER CHUNK ID: %x %x %x %x %x\n", info.ID[0], info.ID[1], info.ID[2], info.ID[3], info.ID[4]);
        return 1;
    }
    readReverse(info.size);
    readReverse(info.formatType);
    readReverse(info.trackCount);
    readReverse(info.timeDivision);
    std::printf("0x%08x; 0x%04x; 0x%04x; 0x%04x;\n", info.size, info.formatType, info.trackCount, info.timeDivision);
    info.dataPosition = file->tellg();
    info.lastPosition = info.dataPosition;

    if (info.formatType != 0){
        std::fprintf(stderr, "ERR MidiFileReader::parseFile: MIDI FILE FORMAT IS NOT EQUAL TO 0\n");
        return 2;
    }

    settings.calculateTickValue(info.timeDivision, sampleRate, sampleSize);

    chunks = new midiChunk[info.trackCount];
    lastEvent = new midiEvent[info.trackCount];
    chunkTime = new double[info.trackCount];
    lastEventTime = new ulong[info.trackCount];
    endOfChunk = new bool[info.trackCount];
    for (uint i = 0; i < info.trackCount; i++){
        chunkTime[i] = 0;
        lastEventTime[i] = 0;
        endOfChunk[i] = false;
    }

    uchar byte;
    char ID[5] = "MTrk";
    for (uint i = 0; i < info.trackCount; i++){
        for (uint j = 0; j < 4;){
            file->get((char&)(byte));
            if (file->eof()){
                std::fprintf(stderr, "ERR MidiFileReader::parseFile: FILE DOES NOT CONTAIN SPECIFIED AMOUNT OF TRACKS (%i, but only %i found)\n", info.trackCount, i);
                return 3;
            }
            if (ID[j] == byte){
                j++;
            } else {
                j = 0;
            }
        }
        strlcpy(chunks[i].ID, ID, 5);
        readReverse(chunks[i].size);
        chunks[i].dataPosition = file->tellg();
        chunks[i].lastPosition = chunks[i].dataPosition;
    }

    return 0;
}

uchar** MidiFileReader::getTempBuffer(){
    return tempNoteBuffer;
}

void MidiFileReader::setObserver(IMidiFileReaderObserver* observer){
    this->observer = observer;
}

void MidiFileReader::notifyObserver(){
    if (endOfChunk[0] == true){
        if (observer != nullptr){
            observer->notifyFileEnd();
        }   
    }
}
