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
pcm_samples =analog_signal.astype(np.float32)
# Drop every 9 per 10 samples in PCM signal
dropped_samples = [x for i, x in enumerate(pcm_samples) if i % 2000 == 1]
# Generate time axis for dropped samples
t_dropped = [x for i, x in enumerate(t) if i % 2000 == 1]

plt.figure(figsize=(6, 2))

# Plot dropped PCM samples and analog signal in one plot
plt.plot(t, analog_signal, label='Sygnał analogowy')
plt.plot(t_dropped, dropped_samples, marker='o', linestyle='', color='r', label='Próbki LPCM')
plt.xlabel('Czas')
plt.ylabel('Amplituda')
# plt.title('Próbkowanie LPCM FLOAT32')
plt.legend(loc='lower left')

# Show the plot
plt.tight_layout()
plt.subplots_adjust(left=0.1)
plt.show()
