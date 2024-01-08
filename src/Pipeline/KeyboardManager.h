#ifndef KEYBOARDMANAGER_H
#define KEYBOARDMANAGER_H

#include "../UserInput/AKeyboardRecorder.h"
#include "../UserInput/keyboardTransferBuffer.h"
#include "IDManager.h"
#include "IDManager.cpp"

#include <cstring>

namespace pipeline{
    template<typename CAPACITY>
    class KeyboardManager: public IDManager<AKeyboardRecorder, CAPACITY>{
    public:
        KeyboardManager(CAPACITY defaultIncrement = 8);
        ~KeyboardManager() /*override*/;

        CAPACITY add(AKeyboardRecorder*& newSynth) override;
        char remove(CAPACITY ID) override;
        void removeAll() override;

        void swapActiveBuffers();
        keyboardTransferBuffer* getBuffer(CAPACITY ID);
        keyboardTransferBuffer** getAllBuffers();

        typedef void (keyboardTransferBuffer::*methodPtr)();
        void doForAllBuffers(methodPtr method);

        using IDManager<AKeyboardRecorder, CAPACITY>::getElement;
        using IDManager<AKeyboardRecorder, CAPACITY>::IDValid;
        using IDManager<AKeyboardRecorder, CAPACITY>::getElementCount;
        using IDManager<AKeyboardRecorder, CAPACITY>::getAll;
        using IDManager<AKeyboardRecorder, CAPACITY>::doForAll;

    private:
        void resizeElements(CAPACITY increment) override;

        keyboardTransferBuffer** keyboardsState;

        using IDManager<AKeyboardRecorder, CAPACITY>::resizeMap;
        using IDManager<AKeyboardRecorder, CAPACITY>::elements;
        using IDManager<AKeyboardRecorder, CAPACITY>::elementsUsed;
        using IDManager<AKeyboardRecorder, CAPACITY>::elementsID;
        using IDManager<AKeyboardRecorder, CAPACITY>::elementsTotal;
        using IDManager<AKeyboardRecorder, CAPACITY>::IDMap;
        using IDManager<AKeyboardRecorder, CAPACITY>::IDMapUsed;
        using IDManager<AKeyboardRecorder, CAPACITY>::IDMapTotal;
        using IDManager<AKeyboardRecorder, CAPACITY>::defaultIncrement;
    };
}



#endif
