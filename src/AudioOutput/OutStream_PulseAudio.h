#ifndef _OUTSTREAMPULSEAUDIO_H
#define _OUTSTREAMPULSEAUDIO_H

#include "IOutStream.h"

class OutStream_PulseAudio : public IOutStream{
public:
    OutStream_PulseAudio();
    ~OutStream_PulseAudio() override;
    char init(const audioFormatInfo& info, const std::string& appName, const std::string& description) override;
    char playBuffer(audioBuffer* buffer) override;
    char flushBuffer() override;
    char drainBuffer() override;
private:
    pa_simple *playbackStream;
    pa_sample_spec sampleSpecs;
};

#endif
