#ifndef ISCRIPTREADER_H
#define ISCRIPTREADER_H

#include <string>

class IScriptReaderClient{
protected:
    friend class ScriptReader;

    virtual void executeCommand(std::string& command) = 0;
    virtual bool getErrorFlag() = 0;
    virtual void clearErrorFlag() = 0;
};

#endif
