from sklearn.decomposition import PCA
from scipy.signal import correlate, find_peaks
import numpy as np
import matplotlib.pyplot as plt

class QRSdetector:
    def __init__(self, stochastic_process, template_duration, threshold_factor, sr):
        """
        Creates a QRS detector: An object capable of detecting the peaks of QRS complexes in a stochastic process related to 
        the most prominent component of the process. For instance, if the signals are primarily composed of the component 
        associated with the maternal ECG, this module can detect the positions of the maternal QRS peaks.

        Args: 
            stochastic_process (StochasticProcess): The process whose QRS peaks need to be detected
            template_duration (int): The duration of the template (seconds) characterizing the average QRS complex of the process
            threshold_factor (float): The threshold to apply to the correlation between the template and the enhanced QRS signal (first principal component)
            sr (int): process sample rate
        
        Raises:
            ValueError if template_duration is invalid (equal or less than zero)
            ValueError if threshold_factor is invalid (major than 1)
        """
        self.stochastic_process = stochastic_process
        self.template_duration = template_duration
        self.threshold_factor = threshold_factor
        self.sr = sr

        if template_duration <= 0: raise ValueError("invalid template duration")
        if threshold_factor > 1: raise ValueError("invalid threshold factor")


    def get_enhanced_QRS(self):
        """
        Returns:
            enhanced_QRS (list): the stochastic process component with the largest variance
            explained_variance_ratio (float): the amount of the total variance explained by the first principal component
        """
        X = np.column_stack([realization for realization in self.stochastic_process])
        
        # Normalization
        norms = np.linalg.norm(X, axis=0)
        X_normalized = X / norms

        # PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_normalized)
        enhanced_QRS = pc1.flatten()
        return enhanced_QRS, pca.explained_variance_ratio_[0]*100
    

    def create_qrs_template(self, enhanced_QRS):
        """
        Creates the QRS template

        Args:
            enhanced_QRS: the first principal component of the process from which the template is created

        Returns:
            final_template: the average QRS template

        Raises:
            ValueError if no QRS template can be extracted from the enhanced QRS signal
        """
        window_length = self.sr  # Number of samples in 1 second, the amount of time that contains at least 1 QRS complex 
        qrs_length = int(self.template_duration * self.sr)
        half_qrs = qrs_length // 2

        num_windows = len(enhanced_QRS) // window_length
        templates = []

        # Templates Extraction from enhanced_QRS
        for i in range(num_windows):
            start_idx = i * window_length
            end_idx = start_idx + window_length
            segment = enhanced_QRS[start_idx:end_idx]

            # index of the absolute maximum (R peak) with respect to the window
            max_idx = np.argmax(np.abs(segment))

            # index of the absolute maximum (R peak) with respect to the whole signal
            global_max_idx = start_idx + max_idx

            if global_max_idx - half_qrs >= 0 and global_max_idx + half_qrs < len(enhanced_QRS):
                template = enhanced_QRS[global_max_idx - half_qrs: global_max_idx + half_qrs]
                templates.append(template)

        # Templates averaging 
        if len(templates) > 0: final_template = np.mean(templates, axis=0)
        else: raise ValueError("Error while creating the QRS template from the enhanced QRS signal")

        return final_template
    

    def detect_qrs(self, enhanced_QRS, template):
        """
        Detect QRS peaks using a cross-correlation + threshold method

        Args:
            enhanced_QRS (array): the first principal component of the process from which the template is created
            template (array): QRS template.

        Returns:
            corrected_peaks (list): Positions (indexes) of the detected QRS complexes 
            cross_corr_norm (array): normalized cross-correlation
        """
        cross_corr = correlate(enhanced_QRS, template, mode='full')
        cross_corr = cross_corr[len(template)-1:]

        # cross-correlation normalization
        norm_factor = (np.linalg.norm(template, axis=0))**2
        cross_corr_norm = cross_corr / norm_factor

        # Peaks detection (above the threshold, which is calculated with respect to the cross correlation maximum value)
        peaks, _ = find_peaks(cross_corr_norm, height=self.threshold_factor*np.max(cross_corr_norm), distance=self.sr//4)

        # a cross-correlation peak is aligned with the beginning of the template (when the template starts to overlap with the signal), not with the centre
        # so the correct peak position half of len(template) samples above
        corrected_peaks = peaks + len(template) // 2

        return corrected_peaks, cross_corr_norm
