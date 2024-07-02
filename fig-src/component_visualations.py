import matplotlib.pyplot as plt
import numpy as np
import math

sample_rate = 2000
duration = 2.0
sample_size = int(sample_rate * duration)
frequency = 3

def generate_buffer():
    t = np.linspace(0, 1, sample_size, False)
    n = np.arange(0, sample_size, 1)
    buffer = np.sin((2 * np.pi * frequency * n) / sample_rate)
    return t, buffer
    

def compressor(buffer):
    out = np.zeros(sample_size)
    level = 0
    threshold = 0.5
    levelRiseTime = 0.25 * sample_rate
    levelDecreaseTime = 1 * sample_rate
    vol = 1
    ratio = 10

    for i in range(sample_size):
        abs = math.fabs(buffer[i])


        if abs >= level:
            level += abs / levelRiseTime
        else:
            level -= abs / levelDecreaseTime

        diff = level - threshold

        if diff > 0:
            out[i] = buffer[i] * (threshold + diff / ratio) / level
        else:
            out[i] = buffer[i]

        out[i] *= vol
    return out, "sygnał skompresowany"


def simple_compressor(buffer):
    out = np.zeros(sample_size)
    threshold = 0.5
    ratio = 5
    vol = 1

    for i in range(sample_size):
        abs = math.fabs(buffer[i])
        if abs > threshold:
            out[i] = (threshold + (abs - threshold) / ratio) * np.sign(buffer[i])
        else:
            out[i] = buffer[i]
        out[i] *= vol
    return out, "sygnał skompresowany"


def volume(buffer):
    out = np.zeros(sample_size)
    vol = 0.5
    out = buffer * vol
    return out, "sygnał ze zmodyfikowaną amplitudą"


def distortion(buffer):
    out = np.zeros(sample_size)
    compress = 0.8
    gain = 1.2
    symmetry = 0.3

    positiveGain = gain * (1 + symmetry)
    negativeGain = gain * (1 - symmetry)

    for i in range(sample_size):
        out[i] = buffer[i]
        if buffer[i] > 0:
            out[i] *= positiveGain
            if out[i] > compress:
                out[i] = compress
            
        else:
            out[i] *= negativeGain
            if out[i] < -compress:
                out[i] = -compress
    return out, "sygnał przesterowany"


def delay(buffer):
    repeats = 2
    fade = 0.5

    out = np.zeros(sample_size * (repeats+1))

    for i in range(repeats+1):
        for j in range(sample_size):
            out[i * sample_size + j] = buffer[j] * fade**i
        

    out = out[::(repeats+1)]
    return out, "sygnał z opóźnieniem"
    

t, buffer = generate_buffer()
modified_buffer, label = simple_compressor(buffer)

# buffer = np.append(buffer, np.zeros(sample_size * 2))
# buffer = buffer[::(3)]

print(buffer.shape, modified_buffer.shape)

plt.figure(figsize=(6, 2))

# Plot dropped PCM samples and analog signal in one plot
plt.plot(t, buffer, label='sygnał wejściowy', marker='.', linestyle='', color='red')
plt.plot(t, modified_buffer, label=label, marker='.', linestyle='', color='blue')
plt.legend(loc='lower left')


# Show the plot
plt.tight_layout()
plt.subplots_adjust(left=0.1)
plt.show()
