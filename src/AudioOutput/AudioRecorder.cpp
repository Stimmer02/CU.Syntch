#include "AudioRecorder.h"

AudioRecorder::AudioRecorder(){
    file = nullptr;
    savedData = 0;
    fileSizePosition = 0;
    dataSizePosition = 0;
}

AudioRecorder::~AudioRecorder(){
    closeFile();
}

void AudioRecorder::writeInverted(uint64_t input, uchar length){
    char writeBuffer[8];
    static const uint64_t mask = 0xFF;
    for (uchar i = 0; i < length; i++){
        writeBuffer[i] = input & mask;
        input >>= 8;
    }

    file->write(writeBuffer, length);
}

char AudioRecorder::init(const audioFormatInfo& info){
    return init(info, "./out.wav");
}

char AudioRecorder::init(const audioFormatInfo& info, std::string fileName){
    if (file != nullptr){
        closeFile();
    }
    file = new std::fstream(fileName, std::ios_base::out | std::ios_base::binary);
    if (file->bad()){
        delete file;
        return 1;
    }

    uint subchunk1Size = 16, audioFormat = 1, empty = 0;

    if (info.littleEndian){
        file->write("RIFF", 4);
    } else {
        file->write("RIFX", 4);
    }
    fileSizePosition = file->tellp();
    file->write(reinterpret_cast<const char *>(&empty), 4);
    file->write("WAVE", 4);
    file->write("fmt ", 4);
    writeInverted(subchunk1Size, 4);
    writeInverted(audioFormat, 2);
    file->write(reinterpret_cast<const char *>(&info.channels), 1);
    file->write(reinterpret_cast<const char *>(&empty), 1);
    writeInverted(info.sampleRate, 4);
    writeInverted(info.byteRate, 4);
    writeInverted(info.blockAlign, 2);
    file->write(reinterpret_cast<const char *>(&info.bitDepth), 1);
    file->write(reinterpret_cast<const char *>(&empty), 1);
    file->write("data", 4);
    dataSizePosition = file->tellp();
    file->write(reinterpret_cast<const char *>(&empty), 4);

    savedData = 0;

    return 0;
}

char AudioRecorder::saveBuffer(const audioBuffer* buffer){
    if (file == nullptr){
        return 1;
    }

    file->write((char*)(buffer->buff), buffer->count);
    savedData += buffer->count;

    return 0;
}

char AudioRecorder::closeFile(){
    if (file == nullptr){
        return 1;
    }
    if (file->is_open() == false){
        return 2;
    }

    file->seekp(dataSizePosition, std::ios_base::beg);
    file->write(reinterpret_cast<const char *>(&savedData), 4);
    savedData += 36;
    file->seekp(fileSizePosition, std::ios_base::beg);
    file->write(reinterpret_cast<const char *>(&savedData), 4);

    file->close();
    delete file;
    file = nullptr;
    return 0;
}
