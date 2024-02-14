#ifndef SCRIPTREADER_H
#define SCRIPTREADER_H

#include "IScriptReaderClient.h"
#include <fstream>

class ScriptReader{
public:
    ScriptReader(IScriptReaderClient* client);

    char executeScript(std::string scriptPath, bool verbose);
    std::string getLastError();

private:
    IScriptReaderClient* client;
    std::string lastError;
};

#endif
