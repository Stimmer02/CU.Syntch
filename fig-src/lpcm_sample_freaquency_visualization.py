import numpy as np

import matplotlib.pyplot as plt


# Parameters
sampling_rate = 44000  # Sample rate (Hz)
duration = 1.0  # Duration of the signal (seconds)
frequency = 1  # Frequency of the sine wave (Hz)

# Generate time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate analog signal
analog_signal = np.sin(2 * np.pi * frequency * t)

# Generate PCM samples
samples_250 = np.array([x for i, x in enumerate(analog_signal) if i % 1760 == 1])
t_250 = [x for i, x in enumerate(t) if i % 1760 == 1]
samples_1k = np.array([x for i, x in enumerate(analog_signal) if i % 440 == 1])
t_1k = [x for i, x in enumerate(t) if i % 440 == 1]
samples_4k = np.array([x for i, x in enumerate(analog_signal) if i % 110 == 1])
t_4k = [x for i, x in enumerate(t) if i % 110 == 1]



# Plot
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(t_250, samples_250, label='LPCM 250Hz', marker='.', linestyle='')
plt.title('LPCM 250Hz')

plt.subplot(3, 1, 2)
plt.plot(t_1k, samples_1k, label='LPCM 1kHz', marker='.', linestyle='')
plt.title('LPCM 1kHz')
plt.ylabel('Amplituda')


plt.subplot(3, 1, 3)
plt.plot(t_4k, samples_4k, label='LPCM 4kHz', marker='.', linestyle='')
plt.title('LPCM 4kHz')
plt.xlabel('Czas (s)')



plt.tight_layout()
plt.show()
