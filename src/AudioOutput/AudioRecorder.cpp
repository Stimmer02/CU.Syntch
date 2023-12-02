#include "AudioRecorder.h"
#include <cstdio>

AudioRecorder::AudioRecorder(){
    file = nullptr;
}

AudioRecorder::~AudioRecorder(){
    closeFile();
}

void AudioRecorder::writeInverted(uint64_t input, uchar length){
    static char writeBuffer[8];
    static const uint64_t mask = 0xFF;
    for (uchar i = 0; i < length; i++){
        writeBuffer[i] = input & mask;
        input >>= 8;
    }

    std::fwrite(writeBuffer, sizeof(char), length, file);
}

char AudioRecorder::init(const audioFormatInfo& info, std::string fileName){
    if (file != nullptr){
        closeFile();
    }
    file = std::fopen(fileName.c_str(), "wb");
    if (std::ferror(file)){
        fclose(file);
        return 1;
    }
    if (initFile(info)){
        return 2;
    }

    return 0;
}

char AudioRecorder::initFile(const audioFormatInfo& info){
    uint subchunk1Size = 16, audioFormat = 1, empty = 0;

    if (info.littleEndian){
        std::fwrite("RIFF", sizeof(char), 4, file);
    } else {
        return 1;
        // file->write("RIFX", 4);
    }
    std::fgetpos(file, &fileSizePosition);
    std::fwrite(&empty, sizeof(uint), 1, file);
    std::fwrite("WAVE", sizeof(char), 4, file);
    std::fwrite("fmt ", sizeof(char), 4, file);
    writeInverted(subchunk1Size, 4);
    writeInverted(audioFormat, 2);
    std::fwrite(&info.channels, sizeof(char), 1, file);
    std::fwrite(&empty, sizeof(char), 1, file);
    writeInverted(info.sampleRate, 4);
    writeInverted(info.byteRate, 4);
    writeInverted(info.blockAlign, 2);
    std::fwrite(&info.bitDepth, sizeof(char), 1, file);
    std::fwrite(&empty, sizeof(char), 1, file);
    std::fwrite("data", sizeof(char), 4, file);
    std::fgetpos(file, &dataSizePosition);
    std::fwrite(&empty, sizeof(uint), 1, file);

    savedData = 0;

    return 0;
}

char AudioRecorder::init(const audioFormatInfo& info){
    if (file != nullptr){
        closeFile();
    }
    freopen(NULL, "wb", stdout);
    file = stdout;
    if (std::ferror(file)){
        fclose(file);
        return 1;
    }
    if (initFile(info)){
        return 2;
    }

    return 0;
}

char AudioRecorder::saveBuffer(const audioBuffer* buffer){
    if (file == nullptr){
        return 1;
    }

    std::fwrite(buffer->buff, sizeof(char), buffer->count, file);
    savedData += buffer->count;

    return 0;
}

char AudioRecorder::closeFile(){
    if (file == nullptr){
        return 1;
    }
    // if (file->is_open() == false){
    //     return 2;
    // }
    std::fsetpos(file, &dataSizePosition);
    std::fwrite(&savedData, sizeof(uint), 1, file);
    savedData += 36;
    std::fsetpos(file, &fileSizePosition);
    std::fwrite(&savedData, sizeof(uint), 1, file);

    if (file != stdout){
        fclose(file);
    }
    // delete file;
    return 0;
}
