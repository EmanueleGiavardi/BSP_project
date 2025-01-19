from scipy.signal import firwin, filtfilt, iirnotch
import numpy as np

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
    

    def remove_PLI_adaptive(self, signal, timestamps, epsilon, max_iterations):
        """
        Attemptive simplified implementation of the adptive filtering approach explained in
        "An Improved Adaptive Power Line Interference Canceller for Electrocardiography" by 
        S.M.M. Martens et al.

        Due to the complexity of this approach and the poor performance of this simplified implementation,
        this is not used to remove PLI
        """
        
        # Initialization values for PLI amplitude (A) and phase (phi)
        A_hat = 3.0 
        phi_hat = 0  

        # Learning Rates for amplitude and phase
        K_A = 0.1  
        K_phi = 0.1  

        pli_estimates = []
        errors = []
        MSEs = []

        for iteration in range(max_iterations):
            filtered_signal = []
            error_sum = 0

            for n in range(len(timestamps)):
                pli_hat = A_hat * np.cos(2 * np.pi * self.PLI_freq * timestamps[n] + phi_hat)

                e = signal[n] - pli_hat
                error_sum += e**2

                # Adaptation Subscheme (updates parameters using gradient descent)
                A_hat += K_A * e * np.cos(2 * np.pi * self.PLI_freq * timestamps[n] + phi_hat)
                phi_hat -= K_phi * e * A_hat * np.sin(2 * np.pi * self.PLI_freq * timestamps[n] + phi_hat)

                # Updated PLI estimate
                pli_hat = A_hat * np.cos(2 * np.pi * self.PLI_freq * timestamps[n] + phi_hat)
                filtered_signal.append(e)
                pli_estimates.append(pli_hat)

            #print(f"estimate PLI amplitude at iteration {iteration}: {A_hat}")
            #print(f"estimate PLI phase at iteration {iteration}: {phi_hat}")
            #print("-------------------------------------------------------\n")

            MSE = error_sum / len(timestamps)
            MSEs.append(MSE)

            # Check stopping criteria
            if iteration > 0 and abs(MSEs[-1] - MSEs[-2]) < epsilon:
                print(f"Convergence reached at iteration {iteration} with MSE: {MSE}")
                break
        
        return np.array(filtered_signal)


        