import numpy as np
import matplotlib.pyplot as plt


sampling_rate = 2000  # Sample rate (Hz)
duration = 1.0  # Duration of the signal (seconds)
frequency = 4  # Frequency of the sine wave (Hz)

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
phase = sampling_rate/frequency/4
n = np.arange(phase, int(sampling_rate * duration) + phase, 1)

# sine = np.sin((2 * np.pi * frequency * n) / sampling_rate)

# square = (((frequency * n * 2) // sampling_rate) & 1) * -2 + 1

# sawtooth = (frequency * n / sampling_rate - frequency * n // sampling_rate) * 2 - 1

triangle = ((frequency * 2 * n / sampling_rate) - (frequency * 2 * n // sampling_rate))
triangle = np.where((np.int32(frequency * 2 * n / sampling_rate) & 1) == 0, triangle * 2 - 1, (1 - triangle) * 2 - 1)

plt.figure(figsize=(6, 2))

# Plot dropped PCM samples and analog signal in one plot
plt.plot(t, triangle, label='triangle wave', marker='.', linestyle='--')

# Show the plot
plt.tight_layout()
# plt.subplots_adjust(left=0.1)
plt.show()
