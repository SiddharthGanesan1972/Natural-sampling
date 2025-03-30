Aim 
To perform Natural Sampling of a continuous - time message signal using a pulse train and reconstruct the signal using a low pass filter.

Tools required 
Python (Version 3.x), Numpy Library (for numerical computation), Matplotlib Library (for plotting the waveforms), Scipy Library (for signal processing, including filtering).

Program 
```import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector

# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)

# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)  # Define width of each pulse
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1

# Natural Sampling
nat_signal = message_signal * pulse_train

# Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]

# Create a time vector for the sampled points
sample_times = t[pulse_train == 1]

# Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))  # Find the closest index to the sample time
    reconstructed_signal[index] = sampled_signal[i]

# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

# Apply low-pass filter to smooth the reconstruction
reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

# Plotting
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```
Output Waveform 
![image](https://github.com/user-attachments/assets/61f65a8a-b7cd-4384-b37b-8fb4683457ea)

Results

The continuous sine wave is sampled at regular intervals, where the sampling signal multiplies the continuous signal, resulting in discrete samples. The output shows how the continuous signal is sampled at specified intervals. These discrete samples represent the continuous signal at the sampling times.
