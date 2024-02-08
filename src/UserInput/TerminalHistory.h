#ifndef TERMINALHISTORY_H
#define TERMINALHISTORY_H

#include <string>
#include <vector>
#include <fstream>

class TerminalHistory{
public:
    TerminalHistory(std::string saveFilePath);
    ~TerminalHistory();

    void addEntry(std::string entry);
    std::string getPreviousEntry();
    std::string getNextEntry();
    void resetIndex();
    void clearHistory();

private:
    uint index;
    std::vector<std::string> history;
    std::string saveFilePath;
};

#endif
