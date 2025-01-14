import numpy as np

class ECGestimator:
    def __init__(self, stochastic_process, P_wave_duration, QRS_complex_duration, T_wave_duration, sr, labels):
        """
        Creates an ECG estimator: An object capable of estimate the ECG complex in a stochastic process related to 
        the most prominent component of the process. For instance, if the signals are primarily composed of the component 
        associated with the maternal ECG, this module can create an estimate of the maternal ECG only.

        Args: 
            stochastic_process (StochasticProcess): The process whose ECG complex need to be estimated
            P_wave_duration (float): duration of the ECG P wave in seconds
            QRS_complex_duration (float) : duration of the QRS complex in seconds
            T_wave_duration (float): duration of the ECG T wave in seconds
            sr (int): process sample rate
            labels (list): list containing names related to this process
        """
        self.stochastic_process = stochastic_process
        self.sr = sr
        self.labels = labels

        self.P_len_samples = int(P_wave_duration * self.sr)
        self.QRS_len_samples = int(QRS_complex_duration * self.sr)
        self.T_len_samples = int(T_wave_duration * self.sr)

        self.samples_before_QRS = (P_wave_duration + QRS_complex_duration/2) * self.sr
        self.samples_after_QRS = (T_wave_duration + QRS_complex_duration/2) * self.sr


    def get_real_ECGs(self, peaks):
        """
        Args: 
            peaks (list): estimated QRS peaks

        Returns:
            real_ECGs (dict): contains a list of the actual K ECG complexes (centered on QRS peaks) for each realization of the process
                real_ECGs["AECGi"] -> [[ECG_i_1], [ECG_i_2], ..., [ECG_i_K]],   for i in (0, num_realizations)
            
            real_ECGs_positions (dict): contains a list of the positions of the actual K ECG complexes (centered on QRS peaks) for each realization of the process
                real_ECGs_positions["AECGi"] -> [(start_ECG_i_1, end__ECG_i_1), ..., (start_ECG_i_K, end__ECG_i_K)], for i in (0, num_realizations)  
        """
        real_ECGs = {}
        real_ECGs_positions = {}

        for i in range(self.stochastic_process.num_realizations):
            signal, realization = self.stochastic_process.get_realization_by_index(i), self.labels[i]
            real_ECGs[realization] = []
            real_ECGs_positions[realization] = []
            for qrs_peak in peaks:
                if (qrs_peak - self.samples_before_QRS > 0 and qrs_peak + self.samples_after_QRS < len(signal)):
                    window = signal[int(qrs_peak - self.samples_before_QRS):int(qrs_peak + self.samples_after_QRS)]
                    real_ECGs[realization].append(window)
                    real_ECGs_positions[realization].append((int(qrs_peak - self.samples_before_QRS), int(qrs_peak + self.samples_after_QRS)))

        return real_ECGs, real_ECGs_positions
    

    def get_ECG_averages(self, real_ECGs):
        """
        Args:
            real_ECGs (dict): contains a list of the actual K ECG complexes (centered on QRS peaks) for each realization of the process
                real_ECGs["AECGi"] -> [[ECG_i_1], [ECG_i_2], ..., [ECG_i_K]],   for i in (0, num_realizations)
            
        Returns:
            ECG_averages (dict): contains the average ECG for each realization 
                ECG_averages["AECGi"] -> mu_i: average ECG of the i-th realization
        """
        ECG_averages = {}

        for label in real_ECGs:
            mu = np.mean(real_ECGs[label], axis=0)
            ECG_averages[label] = mu

        return ECG_averages
    

    def get_mu_portions(self, ECG_averages):
        """
        Args:
            ECG_averages (dict): contains the average ECG for each realization 
                ECG_averages["AECGi] -> mu_i: average ECG of the i-th realization
            
        Returns:
            mu_portions (dict): contains the three portions of the ECG complex (P wave, QRS complex, T wave) for each realization
                mu_portions["AECGi"] -> [ [mu_i_P], [mu_i_QRS], [mu_i_T] ],     for i in (0, num_realizations)
        """
        mu_portions = {}
        for label in ECG_averages:
            mu_P = ECG_averages[label][0:self.P_len_samples]
            mu_QRS = ECG_averages[label][self.P_len_samples:self.QRS_len_samples+self.P_len_samples]
            mu_T = ECG_averages[label][self.QRS_len_samples+self.P_len_samples:]
            mu_portions[label] = [mu_P, mu_QRS, mu_T]
        return mu_portions
    
    
    def matrix_constructor(self, mu_P, mu_QRS, mu_T):
        col_1 = np.concatenate([mu_P, np.zeros(len(mu_QRS) + len(mu_T))])
        col_2 = np.concatenate([np.zeros(len(mu_P)), mu_QRS, np.zeros(len(mu_T))])
        col_3 = np.concatenate([np.zeros(len(mu_P) + len(mu_QRS)), mu_T])
        return np.column_stack([col_1, col_2, col_3])


    def get_M_matrixes(self, mu_portions):
        """
        Args:
            mu_portions (dict): contains the three portions of the ECG complex (P wave, QRS complex, T wave) for each realization
                mu_portions["AECGi"] -> [ [mu_i_P], [mu_i_QRS], [mu_i_T] ],     for i in (0, num_realizations)
        
        Returns:
            M_matrixes (dict): contains the M matrixes related to each realization
        """
        M_matrixes = {}
        for realization in mu_portions: M_matrixes[realization] = self.matrix_constructor(mu_portions[realization][0], 
                                                                                          mu_portions[realization][1], 
                                                                                          mu_portions[realization][2])
        return M_matrixes
    

    def get_optimized_scaled_values(self, M, m):
        """
        Args:
            M (matrix): the M matrix related to a certain realization
            m (array): the true ECG complex
        
        Returns:
            a (array): a three-values array containing the three optimal scaling coefficients for the P wave, QRS complex and T wave
                a[0] = a_P: optimal scaling coefficients for the P wave
                a[1] = a_QRS: optimal scaling coefficients for the QRS complex
                a[2] = a_T: optimal scaling coefficients for the T wave
        """
        MTM_inv = np.linalg.inv(np.dot(M.T, M))
        a = np.dot(MTM_inv, np.dot(M.T, m))
        return a


    def get_scaled_ECG_complex(self, a, mu_P, mu_QRS, mu_T):
        """
        Args:
            a (array): a three-values array containing the three optimal scaling coefficients for the P wave, QRS complex and T wave

        Returns: 
            scaled_mu (array): a three-values array containing the P wave, QRS complex and T wave, each one scaled according to a
        """
        scaled_P = mu_P*a[0]
        scaled_QRS = mu_QRS*a[1]
        scaled_T = mu_T*a[2]
        scaled_mu = np.concatenate([scaled_P, scaled_QRS, scaled_T])
        return scaled_mu
    

    def get_estimated_ECGs(self, real_ECGs, M_matrixes, mu_portions):
        """
        Args: 
            real_ECGs (dict): contains a list of the actual K ECG complexes (centered on QRS peaks) for each realization of the process
                real_ECGs["AECGi"] -> [[ECG_i_1], [ECG_i_2], ..., [ECG_i_K]],   for i in (0, num_realizations)
            M_matrixes (dict): contains the M matrixes related to each realization
            mu_portions (dict): contains the three portions of the ECG complex (P wave, QRS complex, T wave) for each realization
                mu_portions["AECGi"] -> [ [mu_i_P], [mu_i_QRS], [mu_i_T] ],     for i in (0, num_realizations)

        Returns:
            estimated_ECGs (dict): contains a list of the estimated K ECG complexes (centered on QRS peaks) for each realization of the process
        """
        estimated_ECGs = {}
        for label in real_ECGs:
            estimated_ECGs[label] = []
            current_ECGs = real_ECGs[label]
            M = M_matrixes[label]
            K = len(current_ECGs)
            for i in range(K):
                real_complex = current_ECGs[i]
                a = self.get_optimized_scaled_values(M, real_complex)
                mu_P, mu_QRS, mu_T = mu_portions[label][0], mu_portions[label][1], mu_portions[label][2]
                estimated_ECGs[label].append(self.get_scaled_ECG_complex(a, mu_P, mu_QRS, mu_T))

        return estimated_ECGs