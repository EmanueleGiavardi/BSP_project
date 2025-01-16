from scipy.signal import firwin, filtfilt, iirnotch

class ECGcleaner:
    def __init__(self, sr, BW_freq, PLI_freq):
        """
        Creates an ECGcleaner: an object capable of remove or attenutate unwanted frequencies from the ECG 
        (such the ones related to baseline wander or power-line interference) 

        Args:
            sr (int): sample rate of the signal to filter
            BW_freq (float): frequency related to baseline wander
            PLI_freq (float): frequency related to power-line interference
        """
        self.sr = sr
        self.BW_freq = BW_freq
        self.PLI_freq = PLI_freq

    def remove_baseline_wander(self, signal, num_taps):
        """
        Applies a linear-phase FIR high pass filter, with the cutoff frequency specified in the constructor

        Args:
            signal (array): the signal for which the baseline wander has to be removed
        
        Returns:
            filtered_signal (array): the signal without the baseline wander component
        """
        cutoff_normalized = self.BW_freq / (0.5 * self.sr)
        fir_coefficients = firwin(num_taps, cutoff_normalized, pass_zero=False)
        filtered_signal = filtfilt(fir_coefficients, [1.0], signal)

        return filtered_signal
    
    def remove_PLI_notch(self, signal, notch_quality_factor):
        """
        Applies a IIR notch filter, with the cutoff frequency specified in the constructor

        Args:
            signal (array): the signal for which the power-line interference has to be removed
        
        Returns:
            filtered_signal (array): the signal without the PLI component
        """
        b, a = iirnotch(self.PLI_freq, notch_quality_factor, self.sr)
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal


        