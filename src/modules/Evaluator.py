import numpy as np
import matplotlib.pyplot as plt
from modules import StochasticProcess


class Evaluator:
    def __init__(self, S5, gt_S5):
        """
        Creates an Evaluator: an object that implements different methods to evaluate the estimated fetal complexes with respect to the truth ones

        Args: 
            FECG_averages (StochasticProcess): contains the estimated average fetal ECG complexe for each realization of the process
            gt_FECG_averages (StochasticProcess): contains the truth average fetal ECG complexe for each realization of the process
        """
        self.S5 = S5
        self.gt_S5 = gt_S5


    def get_correlations(self):
        """
        Computes the correlation between the estimated average FECG and the estimated truth FECG, for each realization

        Returns:
            correlations (array): correlation values for each realization
            mean_correlation (float): the mean correlation value (on the realizations)
        """
        correlations = []

        for i in range(self.S5.num_realizations):
            correlation = np.corrcoef(self.S5.get_realization_by_index(i), self.gt_S5.get_realization_by_index(i))[0, 1]
            correlations.append(correlation)

        return correlations, np.mean(correlations)

    def get_MSEs(self):
        """
        Computes the Mean Squared Error between the estimated average FECG and the estimated truth FECG, for each realization

        Returns:
            MSEs (array): MSE values for each realization
            mean_MSE (float): the mean MSE value (on the realizations)
        """
        MSEs = []

        for i in range(self.S5.num_realizations):
            MSE = np.mean((self.S5.get_realization_by_index(i) - self.gt_S5.get_realization_by_index(i))**2)
            MSEs.append(MSE)

        return MSEs, np.mean(MSEs)


    def plot_comparison(self, correlations, MSEs):
        """
        Plot the comparison between the average FCEG with estimated and ground truth peaks for each realization, with the associated metrics 

        Args:
            correlations (array): correlation values for each realization
            MSEs (array): MSE values for each realization
        """
        for i in range(self.S5.num_realizations):
            label = f"AECG{i+1}"
            plt.figure(figsize=(12, 8))
            plt.plot(self.S5.get_realization_by_index(i), color='red', label=f"{label}: Average FECG - estimated peaks")
            plt.plot(self.gt_S5.get_realization_by_index(i), color='blue', label=f"{label}: Average FECG - ground truth peaks")

            text_x = 0.05 * len(self.S5.get_realization_by_index(i))
            text_y = max(self.S5.get_realization_by_index(i)) * 0.9
            
            metrics_text = (f"Correlation: {correlations[i]:.4f}\n"
                            f"MSE: {MSEs[i]:.4f}")
            
            plt.text(text_x, text_y, metrics_text, fontsize=11, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            plt.legend()
            plt.show()