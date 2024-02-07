#ifndef INPUTMAP_H
#define INPUTMAP_H

#include <fstream>

typedef unsigned short int ushort;

class InputMap{
public:
    InputMap(ushort keyCount, ushort* map);
    InputMap(std::string path);
    ~InputMap();

    ushort map(const ushort& key)const;
    void replace(ushort& key)const;
    ushort getMaxValue()const;
    ushort getKeyCount()const;

private:
    ushort keyCount;
    ushort* arr;
};

#endif
