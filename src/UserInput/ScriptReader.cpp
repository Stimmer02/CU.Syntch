#include "ScriptReader.h"
#include <string>

ScriptReader::ScriptReader(IScriptReaderClient* client){
    this->client = client;
}

char ScriptReader::executeScript(std::string scriptPath, bool verbose){
    lastError.clear();
    if (client->getErrorFlag()){
        lastError = "ERR ScriptReader::executeScript CLIENT ERROR FLAG SET BEFORE EXECUTION\n";
        return -1;
    }

    std::ifstream script(scriptPath);
    if (script.is_open() == false || script.bad()){
        lastError = "ERR ScriptReader::executeScript COULD NOT READ FILE: " + scriptPath + '\n';
        return -2;
    }

    std::string command;
    for (uint i = 1; std::getline(script, command); i++){
        if (verbose){
            std::printf("\n\e[97m%d: %s\e[0m", i, command.c_str());
        }
        client->executeCommand(command);
        if (client->getErrorFlag()){
            lastError = "ERR ScriptReader::executeScript CLIENT ERROR FLAG SET. FILE:" + scriptPath + "; LINE(" + std::to_string(i) + "):" + command + ";\n";
            return -3;
        }
    }

    script.close();
    return 0;
}

std::string ScriptReader::getLastError(){
    return lastError;
}
