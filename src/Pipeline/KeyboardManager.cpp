#include "KeyboardManager.h"

using namespace pipeline;

template <typename CAPACITY>
KeyboardManager<CAPACITY>::KeyboardManager(CAPACITY defaultIncrement): IDManager<AKeyboardRecorder, CAPACITY>(defaultIncrement){
    keyboardsState = new keyboardTransferBuffer*[elementsTotal];
}

template <typename CAPACITY>
KeyboardManager<CAPACITY>::~KeyboardManager(){
    for (CAPACITY i = 0; i < elementsUsed; i++){
        // delete elements[i];//TODO maybe delete?
        delete keyboardsState[i];
    }

    delete[] keyboardsState;
    // delete[] elements;
    // delete[] elementsID;
    // delete[] IDMap;
}

template <typename CAPACITY>
void KeyboardManager<CAPACITY>::resizeElements(CAPACITY increment){
    elementsTotal += increment;
    AKeyboardRecorder** newElements = new AKeyboardRecorder*[elementsTotal];
    CAPACITY* newElementsID = new CAPACITY[elementsTotal];
    keyboardTransferBuffer** newKeyboardsState = new keyboardTransferBuffer*[elementsTotal];

    std::memcpy(newElements, elements, elementsUsed * sizeof(AKeyboardRecorder*));
    std::memcpy(newElementsID, elementsID, elementsUsed * sizeof(CAPACITY));
    std::memcpy(newKeyboardsState, keyboardsState, elementsUsed * sizeof(keyboardTransferBuffer*));


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
    keyboardsState[elementsUsed] = new keyboardTransferBuffer(newInput->buffer->getSampleSize(), newInput->buffer->getKeyCount());
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
keyboardTransferBuffer* KeyboardManager<CAPACITY>::getBuffer(CAPACITY ID){
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
keyboardTransferBuffer** KeyboardManager<CAPACITY>::getAllBuffers(){
    return keyboardsState;
}
