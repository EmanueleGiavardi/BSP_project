# A Robust Fetal ECG Detection Method for Abdominal Recordings

This project is developed as part of the **Biomedical Signal Processing** course at the **University of Milan**. It implements the fetal ECG detection method proposed by **Suzanna M. M. Martens** in the paper 
*["A Robust Fetal ECG Detection Method for Abdominal Recordings"](https://iopscience.iop.org/article/10.1088/0967-3334/28/4/004/meta)*
## Overview

This project aims to extract the fetal ECG (FECG) from non-invasive abdominal recordings, following the step-by-step approach described in Martens' paper. The method consists of multiple processing stages, including baseline wander removal, powerline interference cancellation, and maternal ECG suppression.

## Notebooks

- **`main.ipynb`**: This notebook provides a step-by-step execution of the separation process. Each module's output is visualized, allowing one to understand how the signal evolves through the different processing stages.
  - You can test the code on a different recording by modifying the filename (e.g., `"a03"`) within the notebook and running all the cells. Valid filenames can be found in the **`data/`** folder.

- **`performances.ipynb`**: This notebook is used for **performance evaluation**. It processes all recordings in the dataset to compute evaluation metrics but does **not** display intermediate outputs of the separation pipeline.

## Dataset

The dataset follows the format used in the **[PhysioNet/Computing in Cardiology Challenge 2013](https://physionet.org/content/challenge-2013/1.0.0/)** for non-invasive fetal ECG extraction. Each recording consists of:
- A **CSV file** containing four abdominal maternal ECG channels.
- A **text file** with the fetal QRS peak locations as ground truth.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/EmanueleGiavardi/BSP_project
   cd BSP_project
   
2. Install dependencies in `requirements.txt`.

3. Open `src/main.ipynb` in Jupyter Notebook or Jupyter Lab.

4. Modify the filename inside the notebook if needed.

5. Run all cells to visualize the processing steps.

6. For dataset-wide evaluation, execute `src/performances.ipynb`.

---

*This project is for academic purposes only*
