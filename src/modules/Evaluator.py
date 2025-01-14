import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, FECG_averages, gt_FECG_averages):
        """
        Creates an Evaluator: an object that implements different methods to evaluate the estimated fetal complexes with respect to the truth ones

        Args: 
            FECG_averages (dict): contains the estimated average fetal ECG complexe for each realization of the process
            gt_FECG_averages (dict): contains the truth average fetal ECG complexe for each realization of the process
        """
        self.FECG_averages = FECG_averages
        self.gt_FECG_averages = gt_FECG_averages


    def get_correlations(self):
        """
        Computes the correlation between the estimated average FECG and the estimated truth FECG, for each realization

        Returns:
            correlations (dict): correlation values for each realization
            mean_correlation (float): the mean correlation value (on the realizations)
        """
        correlations = {}
        mean_correlation = 0

        for label in self.FECG_averages:
            correlation = np.corrcoef(self.FECG_averages[label], self.gt_FECG_averages[label])[0, 1]
            mean_correlation += correlation
            correlations[label] = correlation

        mean_correlation = mean_correlation / len(self.FECG_averages)
        return correlations, mean_correlation

    def get_MSEs(self):
        """
        Computes the Mean Squared Error between the estimated average FECG and the estimated truth FECG, for each realization

        Returns:
            MSEs (dict): MSE values for each realization
            mean_MSE (float): the mean MSE value (on the realizations)
        """
        MSEs = {}
        mean_MSE = 0

        for label in self.FECG_averages:
            MSE = np.mean((self.FECG_averages[label] - self.gt_FECG_averages[label])**2)
            mean_MSE += MSE
            MSEs[label] = MSE
        
        mean_MSE = mean_MSE / len(self.FECG_averages)
        return MSEs, mean_MSE


    def plot_comparison(self, correlations, MSEs):
        """
        Plot the comparison between the average FCEG with estimated and ground truth peaks for each realization, with the associated metrics 

        Args:
            correlations (dict): correlation values for each realization
            MSEs (dict): MSE values for each realization
        """
        for label in self.FECG_averages:
            print(f"\n--------------------------------{label}--------------------------------\n")
            print(f"Correlation value for realization {label} = {correlations[label]}\n")
            print(f"MSE per realizzazione {label} = {MSEs[label]}\n")
            plt.figure(figsize=(12, 8))
            plt.plot(self.FECG_averages[label], color='red', label=f"Average FECG - estimated peaks")
            plt.plot(self.gt_FECG_averages[label], color='blue', label=f"Average FECG - ground truth peaks")
            plt.legend()
            plt.show()