#ifndef TERMINALINPUTDISCARD_H
#define TERMINALINPUTDISCARD_H

#include <termios.h>
#include <cstring>
#include <unistd.h>
#include <ostream>

class TerminalInputDiscard {
public:
    void disableInput();
    void enableInput(bool silent = false);

private:
    void turnEchoOff();
    void turnEchoOn();

    void turnCanonOff();
    void turnCanonOn();

    void discardInputBuffer(bool silent);
    void discardInputLine();
    void setTermiosBit(const int& fd, const tcflag_t& bit, const int& onElseOff);


    struct termios g_terminalSettings;
};

#endif
