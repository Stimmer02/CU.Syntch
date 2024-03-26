#ifndef MIDIREADERMANAGER_H
#define MIDIREADERMANAGER_H

#include "KeyboardRecorder_MidiFile.h"
#include "../../AudioOutput/audioFormatInfo.h"

#include <map>
#include <vector>

namespace MIDI{
    class MidiReaderManager{
    public:
        MidiReaderManager(const audioFormatInfo* audioInfo);
        void rewind();
        char rewind(short inputID);
        bool isMidiFileReader(short);
        char setFile(short inputID, std::string filePath);
        void add(KeyboardRecorder_MidiFile* midiReader, short inputID);
        void remove(short inputID);
        void play();
        char play(short inputID);
        void pause();
        char pause(short inputID);
        std::string getFile(short inputID);
        void printReaders();
        short getCount();

        int getPlayCounter();

    private:
        std::map<short, KeyboardRecorder_MidiFile*> midiReadersMap;
        std::vector<KeyboardRecorder_MidiFile*> midiReaders;
        const audioFormatInfo* audioInfo;

        int playCounter;
    };
}

#endif
