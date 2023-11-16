#ifndef _INPUTMAP_H
#define _INPUTMAP_H

#include <fstream>

typedef unsigned short int ushort;

class InputMap{
public:
    InputMap(ushort keyCount, ushort* map);
    InputMap(std::string path);
    ~InputMap();

    ushort map(const ushort& key);
    void replace(ushort& key);
    ushort getMaxValue();

private:
    ushort keyCount;
    ushort* arr;
};

#endif
