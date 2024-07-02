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

def generate_square():
    n = np.arange(0, sample_size, 1)
    buffer = (((frequency * n * 2) // sample_rate) & 1) * -2 + 1
    return buffer
    

def sum():
    t, buffer = generate_buffer()
    square = generate_square()
    sum = (buffer + square) / 2 

    plt.figure(figsize=(6, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, buffer, label='sine', marker='.', linestyle='', color='red')
    plt.title('sygnał wejściowy 1')

    plt.subplot(3, 1, 2)
    plt.plot(t, square, label='square', marker='.', linestyle='--', color='red')
    plt.title('sygnał wejściowy 2')

    plt.subplot(3, 1, 3)
    plt.plot(t, sum, label='sum', marker='.', linestyle='--', color='blue')
    plt.title('suma sygnałów')

    plt.tight_layout()
    plt.show()

def pan():
    t, buffer = generate_buffer()
    pan = 0.2

    if pan <= 0.5:
        left = buffer 
        right = buffer * pan * 2
    else:
        left = buffer * (2 - pan * 2)
        right = buffer

    plt.figure(figsize=(6, 4))

    plt.subplot(2, 1, 1)
    plt.plot(t, left, label='modified sine', marker='.', linestyle='', color='blue')
    plt.title('sygnał lewy')

    plt.subplot(2, 1, 2)
    plt.plot(t, buffer, label='sine', marker='.', linestyle='', color='red')
    plt.plot(t, right, label='modified sine', marker='.', linestyle='', color='blue')
    plt.title('sygnał prawy')

    plt.tight_layout()
    plt.show()

sum()