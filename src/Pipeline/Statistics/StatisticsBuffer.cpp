#include"StatisticsBuffer.h"

using namespace statistics;
template<typename T>
StatisticsBuffer<T>::StatisticsBuffer(uint size) : size(size){
    arr = new T[size];
    position = 0;
    for (uint i = 0; i < size; i++){
        arr[i] = 0;
    }
}

template<typename T>
StatisticsBuffer<T>::~StatisticsBuffer(){
    delete [] arr;
}

template<typename T>
void StatisticsBuffer<T>::push(const T& value){
    arr[position] = value;
    position++;
    if (position == size){
        position = 0;
    }
}

template<typename T>
double StatisticsBuffer<T>::average(){
    double sum = 0;
    for (uint i = 0; i < size; i++){
        sum += arr[i];
    }
    return sum/size;
}
