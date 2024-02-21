#ifndef SHIFTBUFFER_H
#define SHIFTBUFFER_H


template<typename TYPE>
class ShiftBuffer{
public:
    typedef bool (*compareFunctionPtr)(TYPE& a, TYPE& b);
    ShiftBuffer(int size, compareFunctionPtr equalFuction);
    ~ShiftBuffer();

    void put(const TYPE& newItem);
    void putIfUnique(TYPE& newItem);
    TYPE get();
    int leftInBuffer();

    const int size;
private:
    int count;
    int beginning;
    int end;

    TYPE* arr;

    compareFunctionPtr equalFuction;
};

template<typename TYPE>
ShiftBuffer<TYPE>::ShiftBuffer(int size, compareFunctionPtr equalFuction): size(size){
    this->equalFuction = equalFuction;

    arr = new TYPE[size];
    count = 0;
    beginning = 0;
    end = 0;
}

template<typename TYPE>
ShiftBuffer<TYPE>::~ShiftBuffer(){
    delete[] arr;
}

template<typename TYPE>
void ShiftBuffer<TYPE>::put(const TYPE& newItem){
    arr[end] = newItem;
    end++;
    count++;
    if (end == size){
        end = 0;
    }
}

template<typename TYPE>
void ShiftBuffer<TYPE>::putIfUnique(TYPE& newItem){
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
TYPE ShiftBuffer<TYPE>::get(){
    TYPE& out = arr[beginning];
    beginning++;
    count--;
    if (beginning == size){
        beginning = 0;
    }
    return out;
}

template<typename TYPE>
int ShiftBuffer<TYPE>::leftInBuffer(){
    return count;
}


#endif
