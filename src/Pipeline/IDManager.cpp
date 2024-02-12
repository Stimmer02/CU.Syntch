#include "IDManager.h"

using namespace pipeline;

template <class TYPE, typename CAPACITY>
IDManager<TYPE, CAPACITY>::IDManager(CAPACITY defaultIncrement): defaultIncrement(defaultIncrement){
    elementsTotal = this->defaultIncrement;
    elementsUsed = 0;
    elements = new TYPE*[elementsTotal];
    elementsID = new CAPACITY[elementsTotal];

    IDMapTotal = this->defaultIncrement;
    IDMapUsed = 0;
    IDMap = new CAPACITY[IDMapTotal];

}

template <class TYPE, typename CAPACITY>
IDManager<TYPE, CAPACITY>::~IDManager(){
    for (CAPACITY i = 0; i < elementsUsed; i++){
        delete elements[i];
    }

    delete[] elements;
    delete[] elementsID;
    delete[] IDMap;
}

template <class TYPE, typename CAPACITY>
void IDManager<TYPE, CAPACITY>::resizeElements(CAPACITY increment){
    elementsTotal += increment;
    TYPE** newElements = new TYPE*[elementsTotal];
    CAPACITY* newElementsID = new CAPACITY[elementsTotal];

    std::memcpy(newElements, elements, elementsUsed * sizeof(TYPE*));
    std::memcpy(newElementsID, elementsID, elementsUsed * sizeof(CAPACITY));

    delete[] elements;
    delete[] elementsID;

    elements = newElements;
    elementsID = newElementsID;
}

template <class TYPE, typename CAPACITY>
void IDManager<TYPE, CAPACITY>::resizeMap(CAPACITY increment){
    IDMapTotal += increment;
    CAPACITY* newIDMap = new CAPACITY[IDMapTotal];

    std::memcpy(newIDMap, IDMap, IDMapUsed * sizeof(CAPACITY));

    delete[] IDMap;

    IDMap = newIDMap;
}

template <class TYPE, typename CAPACITY>
CAPACITY IDManager<TYPE, CAPACITY>::add(TYPE*& newElement){
    elements[elementsUsed] = newElement;
    newElement = nullptr;

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

template <class TYPE, typename CAPACITY>
char IDManager<TYPE, CAPACITY>::remove(CAPACITY ID){
    if (IDValid(ID) == false){
        return -1;
    }
    CAPACITY elementPosition = IDMap[ID];
    IDMap[ID] = -1;

    elementsUsed--;
    delete elements[elementPosition];

    if (elementPosition != elementsUsed){
        elements[elementPosition] = elements[elementsUsed];
        IDMap[elementsID[elementsUsed]] = elementPosition;
        elementsID[elementPosition] = elementsID[elementsUsed];
    }

    return 0;
}

template <class TYPE, typename CAPACITY>
void IDManager<TYPE, CAPACITY>::removeAll(){
    for (CAPACITY i = 0; i < elementsUsed; i++){
        delete elements[i];
    }
    elementsUsed = 0;

    delete[] IDMap;
    IDMapUsed = 0;
    IDMapTotal = defaultIncrement;
    IDMap = new CAPACITY[IDMapTotal];
}

template <class TYPE, typename CAPACITY>
void IDManager<TYPE, CAPACITY>::reorganizeIDs(){
    delete[] IDMap;
    IDMapUsed = elementsUsed;
    IDMapTotal = IDMapUsed + defaultIncrement;
    IDMap = new CAPACITY[IDMapTotal];

    for (CAPACITY i = 0; i < IDMapUsed; i++){
        IDMap[i] = i;
        elementsID[i] = i;
    }
}

template<class TYPE, typename CAPACITY>
CAPACITY IDManager<TYPE, CAPACITY>::getElementCount(){
    return elementsUsed;
}

template<class TYPE, typename CAPACITY>
TYPE** IDManager<TYPE, CAPACITY>::getAll(){
    return elements;
}

template <class TYPE, typename CAPACITY>
TYPE* IDManager<TYPE, CAPACITY>::getElement(CAPACITY ID){
    return elements[IDMap[ID]];
}

template <class TYPE, typename CAPACITY>
bool IDManager<TYPE, CAPACITY>::IDValid(CAPACITY ID){
    if (ID > IDMapUsed){
        return false;
    }
    return IDMap[ID] >= 0;
}

template <class TYPE, typename CAPACITY>
void IDManager<TYPE, CAPACITY>::doForAll(methodPtr method){
    for (CAPACITY i = 0; i < elementsUsed; i++){
        (elements[i].*method)();
    }
}

template <class TYPE, typename CAPACITY>
TYPE* IDManager<TYPE, CAPACITY>::getElementByIndex(CAPACITY index){
    return elements[IDMap[index]];
}

