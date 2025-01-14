import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import resample

class StochasticProcess:
    def __init__(self, num_realizations, realizations, names, colors, sr):
        self.num_realizations = num_realizations
        self.realizations = realizations
        self.names = names
        self.colors = colors
        self.sr = sr 

        same_length = all(len(r) == len(self.realizations[0]) for r in self.realizations)
        if not same_length:
            raise(ValueError("Not all the realizations have the same length"))
        
        self.duration = int(self.sr/len(self.realizations[0]))
        self.timestamps = np.linspace(0, self.duration, len(self.realizations[0]), endpoint=False)

    def get_duration(self):
        return self.duration
    
    def get_timestamps(self):
        return self.timestamps

    def get_realization_by_index(self, index):
        return self.realizations[index].copy()
    
    def plot_realization(self, index):
        realization = self.realizations[index]
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, realization)
        plt.title(self.names[index])
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_realization_spectrum(self, index, xlim, ylim):
        realization = self.realizations[index]
        K = len(realization)
        freqs = np.linspace(0, self.sr / 2, K // 2)
        fft_values = fft(realization)
        magnitude = np.abs(fft_values[:K // 2])

        plt.figure(figsize=(14, 6))
        plt.plot(freqs, magnitude)
        plt.title(f"{self.names[index]} spectrum (magnitude)")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.grid()
        plt.show()


    def get_realization_by_name(self, name):
        index = -1
        for i,n in enumerate(self.names):
            if n == name: 
                index = i
                break
        if index != -1:
            return self.realizations[index].copy
        else:
            raise(ValueError(f"No realization in stochastic process named {name}"))
        
    def plot(self):
        fig, axes = plt.subplots(self.num_realizations, 1, figsize=(10, 10), sharex=True)
        for i in range(self.num_realizations):
            axes[i].plot(self.timestamps, self.realizations[i], color=self.colors[i])
            axes[i].set_title(self.names[i])
            plt.xlabel("Time [s]")
            plt.tight_layout()
            plt.show()

    def resample_process(self, target_sr):
        resampled_realizations = []
        for realization in self.realizations:
            num_samples = int(len(realization) * target_sr / self.sr)
            resampled_signal = resample(realization, num_samples)
            resampled_realizations.append(resampled_signal)
        
        resampled_SP = self.__class__(self.num_realizations, resampled_realizations, self.names, self.colors, target_sr)
        return resampled_SP.copy()