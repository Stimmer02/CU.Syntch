#ifndef UNIQUESHIFTBUFFER_H
#define UNIQUESHIFTBUFFER_H


template<typename TYPE>
class UniqueShiftBuffer{
public:
    typedef bool (*compareFunctionPtr)(TYPE& a, TYPE& b);
    UniqueShiftBuffer(int size, compareFunctionPtr equalFuction);
    ~UniqueShiftBuffer();

    void putOrMoveToEnd(TYPE& newItem);
    void putIfUnique(TYPE& newItem);
    TYPE get();
    int leftInBuffer();
    void erase(int index);

    const int size;
private:
    int count;
    int beginning;
    int end;

    TYPE* arr;

    void put(const TYPE& newItem);
    compareFunctionPtr equalFuction;
};

template<typename TYPE>
UniqueShiftBuffer<TYPE>::UniqueShiftBuffer(int size, compareFunctionPtr equalFuction): size(size){
    this->equalFuction = equalFuction;

    arr = new TYPE[size];
    count = 0;
    beginning = 0;
    end = 0;
}

template<typename TYPE>
UniqueShiftBuffer<TYPE>::~UniqueShiftBuffer(){
    delete[] arr;
}

template<typename TYPE>
void UniqueShiftBuffer<TYPE>::put(const TYPE& newItem){
    arr[end] = newItem;
    end++;
    count++;
    if (end == size){
        end = 0;
    }
}

template<typename TYPE>
void UniqueShiftBuffer<TYPE>::erase(int index){
    int movedEnd = end + 1;
    if (movedEnd == size){
        movedEnd = 0;
    }
    for (int i = index + 1; i != movedEnd; i++){
        if (i == size){
            i = 0;
            arr[size - 1] = arr[i];
        } else {
            arr[i - 1] = arr[i];
        }
    }

    if (end == 0){
        end = size - 1;
    } else {
        end--;
    }
    count--;
}

template<typename TYPE>
void UniqueShiftBuffer<TYPE>::putOrMoveToEnd(TYPE& newItem){
    for (int i = beginning; i != end; i++){
        if (i == size){
            i = 0;
        }
        if (equalFuction(arr[i], newItem)){
            erase(i);
            break;
        }
    }
    put(newItem);
}

template<typename TYPE>
void UniqueShiftBuffer<TYPE>::putIfUnique(TYPE& newItem){
    for (int i = beginning; i != end; i++){
        if (i == size){
            i = 0;
        }
        if (equalFuction(arr[i], newItem)){
            return;
        }
    }
    put(newItem);
}

template<typename TYPE>
TYPE UniqueShiftBuffer<TYPE>::get(){
    TYPE& out = arr[beginning];
    beginning++;
    count--;
    if (beginning == size){
        beginning = 0;
    }
    return out;
}

template<typename TYPE>
int UniqueShiftBuffer<TYPE>::leftInBuffer(){
    return count;
}


#endif
