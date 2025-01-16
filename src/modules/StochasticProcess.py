import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import resample, welch

class StochasticProcess:
    def __init__(self, num_realizations, realizations, labels, colors, sr):
        """
        Creates a new stochastic process

        Args:
            num_realizations (int): total number of channels
            realizations (array): the actual signals related to this process
            labels (list): list containing names related to this process
            colors (list): list containing colors related to this process (for visual representation only)
            sr (int): signals sample rate
        
        Raises:
            ValueError if realizations have different length
        """
        self.num_realizations = num_realizations
        self.realizations = realizations
        self.labels = labels
        self.colors = colors
        self.sr = sr 

        same_length = all(len(r) == len(self.realizations[0]) for r in self.realizations)
        if not same_length: raise(ValueError("Not all the realizations have the same length"))
        
        self.duration = int(len(self.realizations[0])/self.sr)
        self.timestamps = np.linspace(0, self.duration, len(self.realizations[0]), endpoint=False)


    def __iter__(self):
        """
        Iterates over the realizations of this process
        """
        return iter(self.realizations)


    def get_realization_by_index(self, index):
        """
        Args:
            index (int): index of the channel to extract

        Returns: 
            realization (array): the realization related to the index 

        Raises: 
            ValueError if the index is not valid (outside range [0 ; num_realizations-1])
        """
        if index < self.num_realizations:
            return self.realizations[index].copy()
        else: 
            raise ValueError(f"This process has {self.num_realizations} realizations")
    
    
    def get_realization_by_label(self, label):
        """
        Args:
            label (string): label of the channel to extract

        Returns: 
            realization (array): the realization related to the label 

        Raises: 
            ValueError if no realization is named as the label arg
        """
        index = -1
        for i,n in enumerate(self.labels):
            if n == label: 
                index = i
                break
        if index != -1:
            return self.realizations[index].copy
        else:
            raise(ValueError(f"No realization in stochastic process named {label}"))


    def plot_realization(self, index, title):
        """
        Args:
            index (int): label of the channel to plot
            title (str): plot title

        Raises: 
            ValueError if the index is not valid (outside range [0 ; num_realizations-1])
        """
        if index < self.num_realizations: 
            realization = self.realizations[index]
        else:
            raise ValueError(f"This process has {self.num_realizations} realizations")
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, realization)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


    def plot_realization_spectrum(self, index, title, xlim=100, ylim=100000, plot_magnitude=True):
        """
        Args:
            index (int): label of the channel' spectrum to plot
            plot_magnitude (bool): True for plotting channel's magnitude, False for plotting channel's phase
            title (str): plot title

        Raises: 
            ValueError if the index is not valid (outside range [0 ; num_realizations-1])
        """
        if index < self.num_realizations: 
            realization = self.realizations[index]
        else:
            raise ValueError(f"This process has {self.num_realizations} realizations")
        K = len(realization)
        freqs = np.linspace(0, self.sr / 2, K // 2)
        fft_values = fft(realization)
        magnitude = np.abs(fft_values[:K // 2])
        phase = np.angle(fft_values[:K // 2])
        plt.figure(figsize=(14, 6))

        if plot_magnitude: 
            plt.plot(freqs, magnitude)
        else: 
            plt.plot(freqs, phase)
        
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.grid()
        plt.show()

    def plot_realization_PSD(self, index, segment_length, title):
        frequencies, psd = welch(self.get_realization_by_index(index), self.sr, nperseg=segment_length)

        plt.figure(figsize=(14, 6))
        plt.semilogy(frequencies, psd, label="PSD (Welch)")
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend()
        plt.show()


        
    def plot(self):
        """
        Plots the entire stochastic process, using colors attribute for different channels 
        """
        _, axes = plt.subplots(self.num_realizations, 1, figsize=(12, 12), sharex=True)
        for i in range(self.num_realizations):
            axes[i].plot(self.timestamps, self.realizations[i], color=self.colors[i])
            axes[i].set_title(self.labels[i])
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show()


    def resample_process(self, target_sr):
        """
        Creates a new, resampled stochastic process
        Args:
            target_sr (int): new sample rate

        Returns: 
            resampled_SP: the resampled stochastic process
        """
        resampled_realizations = []
        for realization in self.realizations:
            num_samples = int(len(realization) * target_sr / self.sr)
            resampled_signal = resample(realization, num_samples)
            resampled_realizations.append(resampled_signal)
        
        return self.__class__(self.num_realizations, resampled_realizations, self.labels, self.colors, target_sr)