#ifndef COMPONENTSETTINGS_H
#define COMPONENTSETTINGS_H

#include <string>

typedef unsigned int uint;

namespace pipeline{
    struct componentSettings{
    public:
        componentSettings(const uint count, const std::string* names): count(count), names(names){
            values = new float[count];
        };
        ~componentSettings(){
            delete[] values;
        }

        const uint count;
        float* values;
        const std::string* names;
    };
}

#endif
