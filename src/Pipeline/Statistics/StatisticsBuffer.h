#ifndef STATISTICSBUFFER_H
#define STATISTICSBUFFER_H

typedef unsigned int uint;

namespace statistics{
    template<typename T>
    class StatisticsBuffer{
    public:
        StatisticsBuffer(uint size) : size(size){
            arr = new T[size];
            position = 0;
            for (uint i = 0; i < size; i++){
                arr[i] = 0;
            }
        }
        ~StatisticsBuffer(){
            delete [] arr;
        }

        void push(const T& value){
            arr[position] = value;
            position++;
            if (position == size){
                position = 0;
            }
        }
        double average(){
            double sum = 0;
            for (uint i = 0; i < size; i++){
                sum += arr[i];
            }
            return sum/size;
        }

    private:
        const uint size;
        uint position;
        T* arr;
    };
}

#endif
