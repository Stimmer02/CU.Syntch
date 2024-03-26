#ifndef IMIDIFILEREADEROBSERVER_H
#define IMIDIFILEREADEROBSERVER_H

namespace MIDI{
    class IMidiFileReaderObserver{
        friend class MidiFileReader;
    protected:
        virtual void notifyFileEnd() = 0;
    };
    
}


#endif