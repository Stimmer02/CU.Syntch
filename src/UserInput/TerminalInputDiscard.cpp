#include "TerminalInputDiscard.h"

void TerminalInputDiscard::disableInput(){
    turnEchoOff();
    turnCanonOff();
}

void TerminalInputDiscard::enableInput(bool silent){
    discardInputBuffer(silent);
    turnCanonOn();
    turnEchoOn();
}

void TerminalInputDiscard::turnEchoOff(){
    setTermiosBit(0,ECHO,0);
}

void TerminalInputDiscard::turnEchoOn(){
    setTermiosBit(0,ECHO,1);
}

void TerminalInputDiscard::turnCanonOff(){
    setTermiosBit(0,ICANON,0);
}

void TerminalInputDiscard::turnCanonOn(){
    setTermiosBit(0,ICANON,1);
}

void TerminalInputDiscard::setTermiosBit(const int& fd, const tcflag_t& bit, const int& onElseOff){
    static int first = 1;
    if (first) {
        first = 0;
        tcgetattr(fd,&g_terminalSettings);
    }
    if (onElseOff)
        g_terminalSettings.c_lflag |= bit;
    else
        g_terminalSettings.c_lflag &= ~bit;
    tcsetattr(fd,TCSANOW,&g_terminalSettings);
}

void TerminalInputDiscard::discardInputBuffer(bool silent){
    struct timeval tv;
    fd_set rfds;
    while (1) {
        FD_ZERO(&rfds);
        FD_SET(0,&rfds);
        tv.tv_sec = 0;
        tv.tv_usec = 0;
        if (select(1,&rfds,0,0,&tv) == -1) { fprintf(stderr, "[error] select() failed: %s", strerror(errno) ); exit(1); }
        if (!FD_ISSET(0,&rfds)) break;

        char buf[500];
        ssize_t numRead = read(0,buf,500);
        if (numRead == -1) { fprintf(stderr, "[error] read() failed: %s", strerror(errno) ); exit(1); }
        if (silent == false){
            printf("[debug] cleared %zd chars\n",numRead);
        }
    }
}

void TerminalInputDiscard::discardInputLine(){
    int c;
    while ((c = getchar()) != EOF && c != '\n');
}


