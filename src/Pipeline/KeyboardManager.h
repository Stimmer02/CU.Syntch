#ifndef KEYBOARDMANAGER_H
#define KEYBOARDMANAGER_H

#include "../UserInput/AKeyboardRecorder.h"
#include "../UserInput/keyboardTransferBuffer_CUDA.h"
#include "IDManager.h"

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
        keyboardTransferBuffer_CUDA* getBuffer(CAPACITY ID);
        keyboardTransferBuffer_CUDA** getAllBuffers();

        typedef void (keyboardTransferBuffer_CUDA::*methodPtr)();
        void doForAllBuffers(methodPtr method);

        using IDManager<AKeyboardRecorder, CAPACITY>::getElement;
        using IDManager<AKeyboardRecorder, CAPACITY>::IDValid;
        using IDManager<AKeyboardRecorder, CAPACITY>::getElementCount;
        using IDManager<AKeyboardRecorder, CAPACITY>::getAll;
        using IDManager<AKeyboardRecorder, CAPACITY>::doForAll;
        using IDManager<AKeyboardRecorder, CAPACITY>::getElementByIndex;

    private:
        void resizeElements(CAPACITY increment) override;

        keyboardTransferBuffer_CUDA** keyboardsState;

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

    template <typename CAPACITY>
    KeyboardManager<CAPACITY>::KeyboardManager(CAPACITY defaultIncrement): IDManager<AKeyboardRecorder, CAPACITY>(defaultIncrement){
        keyboardsState = new keyboardTransferBuffer_CUDA*[elementsTotal];
    }

    template <typename CAPACITY>
    KeyboardManager<CAPACITY>::~KeyboardManager(){
        for (CAPACITY i = 0; i < elementsUsed; i++){
            delete keyboardsState[i];
        }
        delete[] keyboardsState;
    }

    template <typename CAPACITY>
    void KeyboardManager<CAPACITY>::resizeElements(CAPACITY increment){
        elementsTotal += increment;
        AKeyboardRecorder** newElements = new AKeyboardRecorder*[elementsTotal];
        CAPACITY* newElementsID = new CAPACITY[elementsTotal];
        keyboardTransferBuffer_CUDA** newKeyboardsState = new keyboardTransferBuffer_CUDA*[elementsTotal];

        std::memcpy(newElements, elements, elementsUsed * sizeof(AKeyboardRecorder*));
        std::memcpy(newElementsID, elementsID, elementsUsed * sizeof(CAPACITY));
        std::memcpy(newKeyboardsState, keyboardsState, elementsUsed * sizeof(keyboardTransferBuffer_CUDA*));


        delete[] elements;
        delete[] elementsID;
        delete[] keyboardsState;

        elements = newElements;
        elementsID = newElementsID;
        keyboardsState = newKeyboardsState;
    }

    template <typename CAPACITY>
    CAPACITY KeyboardManager<CAPACITY>::add(AKeyboardRecorder*& newInput){
        elements[elementsUsed] = newInput;
        keyboardsState[elementsUsed] = new keyboardTransferBuffer_CUDA(newInput->buffer->getSampleSize(), newInput->buffer->getKeyCount());
        newInput = nullptr;

        IDMap[IDMapUsed] = elementsUsed;
        elementsID[elementsUsed] = IDMapUsed;

        elementsUsed++;
        if (elementsUsed == elementsTotal){
            resizeElements(defaultIncrement);
        }

        IDMapUsed++;
        if (IDMapUsed == IDMapTotal){
            resizeMap(defaultIncrement);
        }

        return IDMapUsed-1;
    }

    template <typename CAPACITY>
    char KeyboardManager<CAPACITY>::remove(CAPACITY ID){
        if (IDValid(ID) == false){
            return -1;
        }
        CAPACITY elementPosition = IDMap[ID];
        IDMap[ID] = -1;

        elementsUsed--;
        delete elements[elementPosition];
        delete keyboardsState[elementPosition];

        if (elementPosition != elementsUsed){
            elements[elementPosition] = elements[elementsUsed];
            keyboardsState[elementPosition] = keyboardsState[elementsUsed];
            IDMap[elementsID[elementsUsed]] = elementPosition;
            elementsID[elementPosition] = elementsID[elementsUsed];
        }

        return 0;
    }

    template <typename CAPACITY>
    void KeyboardManager<CAPACITY>::removeAll(){
        for (CAPACITY i = 0; i < elementsUsed; i++){
            delete elements[i];
            delete keyboardsState[i];
        }
        elementsUsed = 0;

        delete[] IDMap;
        IDMapUsed = 0;
        IDMapTotal = defaultIncrement;
        IDMap = new CAPACITY[IDMapTotal];
    }

    template <typename CAPACITY>
    keyboardTransferBuffer_CUDA* KeyboardManager<CAPACITY>::getBuffer(CAPACITY ID){
        return keyboardsState[IDMap[ID]];
    }

    template <typename CAPACITY>
    void KeyboardManager<CAPACITY>::doForAllBuffers(methodPtr method){
        for (CAPACITY i = 0; i < elementsUsed; i++){
            (keyboardsState[i].*method)();
        }
    }

    template <typename CAPACITY>
    void KeyboardManager<CAPACITY>::swapActiveBuffers(){
        for (CAPACITY i = 0; i < elementsUsed; i++){
            elements[i]->buffer->swapActiveBuffer();
        }
    }

    template <typename CAPACITY>
    keyboardTransferBuffer_CUDA** KeyboardManager<CAPACITY>::getAllBuffers(){
        return keyboardsState;
    }
}



#endif
