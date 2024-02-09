#ifndef TERMINALINPUTDISCARD_H
#define TERMINALINPUTDISCARD_H

#include <termios.h>
#include <cstring>
#include <unistd.h>
#include <ostream>
#include <iostream>
#include <sstream>



class TerminalInputDiscard {
public:
    TerminalInputDiscard();

    void disableInput();
    void enableInput(bool silent = false);
    void discardInputBuffer(bool silent);

    void turnStdinOff();
    void turnStdinOn();
private:
    void turnEchoOff();
    void turnEchoOn();

    void turnCanonOff();
    void turnCanonOn();


    void discardInputBuffer2(bool silent);
    void setTermiosBit(const int& fd, const tcflag_t& bit, const int& onElseOff);


    struct termios g_terminalSettings;
    std::streambuf* originalCin;
    std::istringstream* inputBuffer;
};

#endif
