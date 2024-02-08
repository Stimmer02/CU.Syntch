#include "TerminalHistory.h"

TerminalHistory::TerminalHistory(std::string saveFilePath){
    this->saveFilePath = saveFilePath;
    std::ifstream saveFile(saveFilePath);
    if (saveFile.eof()){
        std::printf("WARNING: TerminalHistory::TerminalHistory SAVE FILE IS EMPTY %s\n", saveFilePath.c_str());
        index = 0;
        return;
    }
    if (saveFile.is_open() == false){
        std::printf("ERR: TerminalHistory::TerminalHistory COULD NOT READ FILE %s\n", saveFilePath.c_str());
        index = 0;
        return;
    }
    std::string line;
    while (std::getline(saveFile, line)){
        history.push_back(line);
    }
    saveFile.close();
    index = history.size();
}

TerminalHistory::~TerminalHistory(){
    std::ofstream saveFile(saveFilePath, std::ios::trunc);
    if (saveFile.is_open() == false){
        std::printf("ERR: TerminalHistory::~TerminalHistory COULD NOT SAVE HISTORY IN FILE %s\n", saveFilePath.c_str());
        return;
    }
    for (uint i = 0; i < history.size(); i++){
        std::string& entry = history.at(i);
        saveFile.write(entry.c_str(), entry.size());
        saveFile.put('\n');
    }
    saveFile.close();
}

void TerminalHistory::addEntry(std::string entry){
    index++;
    history.push_back(entry);
}

std::string TerminalHistory::getPreviousEntry(){
    if (index == 0){
        return "";
    } if (index == 1){
        return history.at(0);
    }
    return history.at(--index);
}

std::string TerminalHistory::getNextEntry(){
    if (index == history.size()){
        return history.at(index-1);
    }
    return history.at(index++);
}

void TerminalHistory::resetIndex(){
    index = history.size();
}

void TerminalHistory::clearHistory(){
    history.clear();
    index = 0;
}
