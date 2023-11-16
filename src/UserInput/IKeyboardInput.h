#ifndef _IKEYBOARD_INPUT
#define _IKEYBOARD_INPUT

#include <thread>
#include <string>

typedef unsigned int uint;
typedef unsigned char uchar;


class IKeyboardInput {
public:
    virtual ~IKeyboardInput(){};
    virtual char init(const std::string path) = 0;
    virtual char start() = 0;
    virtual char stop() = 0;
    virtual void getPressedKeys(ushort* keyArr, ushort& count) = 0;
    virtual bool getKeyState(ushort key) = 0;
};

#endif
