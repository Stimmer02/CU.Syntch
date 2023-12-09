#ifndef IKEYBOARD_INPUT
#define IKEYBOARD_INPUT

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
    virtual bool getKeyState(ushort key) = 0;
    virtual const ushort* getPressedKeysArr() = 0;
    virtual ushort getPressedKeysCount() = 0;
    virtual ushort getKeyCount() = 0;
};

#endif
