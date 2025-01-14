from scipy.signal import firwin, filtfilt

class BaselineWanderRemover:
    def __init__(self, sr, cutoff, num_taps):
        """
        Creates a Baseline Wander Remover: an object capable of attenutate the signal's low frequency components
        due to patient's movement or interferences by electrods. 
        This object is implemented using a FIR Linear Phase High Pass filter  

        Args:
            sr (int): sample rate of the signal to filter
            cutoff (float): high pass filter cutoff frequency
            num_taps (int): filter reponse length
        """
        self.sr = sr
        self.cutoff = cutoff
        self.num_taps = num_taps
        self.cutoff_normalized = self.cutoff / (0.5 * self.sr)

    def highpass_fir_filter(self, signal):
        """
        Args:
            signal (list): the signal to which the filter has to be applied
        """
        fir_coefficients = firwin(self.num_taps, self.cutoff_normalized, pass_zero=False)
        filtered_signal = filtfilt(fir_coefficients, [1.0], signal)

        return filtered_signal


        