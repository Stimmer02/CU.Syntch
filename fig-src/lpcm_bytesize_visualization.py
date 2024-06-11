import numpy as np

import matplotlib.pyplot as plt


# Parameters
sampling_rate = 44100  # Sample rate (Hz)
duration = 1.0  # Duration of the signal (seconds)
frequency = 1  # Frequency of the sine wave (Hz)

# Generate time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate analog signal
analog_signal = np.sin(2 * np.pi * frequency * t)

# Generate PCM samples
dropped_samples = np.array([x for i, x in enumerate(analog_signal) if i % 400 == 1])
pcm_samples_int2 = (((dropped_samples+1)/2 * 4 % 4)).astype(np.uint8)
pcm_samples_int4 = (((dropped_samples+1)/2 * 16 % 16)).astype(np.uint8)
pcm_samples_int8 = ((dropped_samples+1)/2 * 256).astype(np.uint8)
# pcm_samples_int16 = (dropped_samples * 32767).astype(np.int16)
t_dropped = [x for i, x in enumerate(t) if i % 400 == 1]


# Plot
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(t_dropped, pcm_samples_int2, label='PCM 2-bit', marker='.', linestyle='')
plt.title('LPCM 2-bit')

plt.subplot(3, 1, 2)
plt.plot(t_dropped, pcm_samples_int4, label='PCM 4-bit', marker='.', linestyle='')
plt.title('LPCM 4-bit')
plt.ylabel('Poziomy kwantyzacji')


plt.subplot(3, 1, 3)
plt.plot(t_dropped, pcm_samples_int8, label='PCM 8-bit', marker='.', linestyle='')
plt.title('LPCM 8-bit')
plt.xlabel('Czas')



plt.tight_layout()
plt.show()
