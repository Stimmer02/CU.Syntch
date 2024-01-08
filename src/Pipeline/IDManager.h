#ifndef IDMANAGER_H
#define IDMANAGER_H

#include <cstring>

namespace pipeline{
    template <class TYPE, typename CAPACITY>
    class IDManager{
    public:
        IDManager(CAPACITY defaultIncrement = 8);
        /*virtual */~IDManager();

        virtual CAPACITY add(TYPE*& newElement);
        virtual char remove(CAPACITY ID);
        virtual void removeAll();
        void reorganizeIDs();

        TYPE* getElement(CAPACITY ID);
        TYPE** getAll();
        CAPACITY getElementCount();
        bool IDValid(CAPACITY ID);

        typedef void (TYPE::*methodPtr)();
        void doForAll(methodPtr method);

    protected:
        TYPE** elements;
        CAPACITY* elementsID;
        CAPACITY elementsUsed;
        CAPACITY elementsTotal;

        CAPACITY* IDMap;
        CAPACITY IDMapUsed;
        CAPACITY IDMapTotal;

        const CAPACITY defaultIncrement;

        virtual void resizeElements(CAPACITY increment);
        void resizeMap(CAPACITY increment);
    };

    enum ID_type{
        INPUT,
        SYNTH,
        COMP,
    };
}

#endif
