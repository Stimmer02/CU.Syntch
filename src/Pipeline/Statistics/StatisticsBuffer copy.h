#ifndef _STATISTICSBUFFER_H
#define _STATISTICSBUFFER_H

typedef unsigned int uint;

namespace statistics{
    template<typename T>
    class StatisticsBuffer{
    public:
        StatisticsBuffer(uint size);
        ~StatisticsBuffer();

        void push(const T& value);
        double average();

    private:
        const uint size;
        uint position;
        T* arr;
    };
}

#endif
