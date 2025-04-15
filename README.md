# FinRL_Crypto – Enhanced Fork for Rapid Experiments in DRL Cryptocurrency Trading

---
## CS 5720 (CRN:22286) Mini Project
### Student Information
- **Name:** Nidhin Ninan
- **Course:** CS5720 - Neural Networks and Deep Learning
- **Term:** Spring 2025
- **University:** University of Central missouri

### Code Implementation Summary Powerpoint: [NNDL_miniPrj-codePresentation](https://ucmo0-my.sharepoint.com/:p:/g/personal/nxn24130_ucmo_edu/Ec0edLLm1TZDmrbqdxFP4HgBR3BgKyfaH7NFe044Ls7KAw?e=sNMfMC)
### Paper Summary Powerpoint : [PaperSummary_Presentation_NNDL_miniPrj](https://ucmo0-my.sharepoint.com/:p:/g/personal/nxn24130_ucmo_edu/EXiALHDbcJRHrAtMZMeLMv0BdE5HSJQjnlNMefLc47LwXg?e=9V40Fn)
### Mini-Project Paper Summary : [Summary_Project_NNDL_NNinan](https://ucmo0-my.sharepoint.com/:w:/g/personal/nxn24130_ucmo_edu/EZGlC4Eo171FpJQsepWebxcBF7LM5M34FwmvY56Hdiu9Vg?e=gfc413)
---

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository is a fork of the original [FinRL_Crypto](https://github.com/AI4Finance-Foundation/FinRL_Crypto) project with important modifications that enable rapid experimentation and free-run demonstration on platforms such as Google Colab and Kaggle Notebook.

> **Note:** Due to limitations imposed by free GPU runtimes, this fork significantly reduces the number of tickers, the number of candles used for training/testing, and the number of Optuna trials. Consequently, while our experiments run in a fraction of the original time (< 25 hrs), the quality of the trained models is correspondingly diminished. The modifications were performed primarily in the configuration files (`config_main`) and within the hyperparameter sweep scripts (namely, the `1_optimize_$.py` files).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Experiment Details](#experiment-details)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Cross-Validation Methods](#cross-validation-methods)
- [Results and Analysis](#results-and-analysis)
  - [Quality of Experiment](#quality-of-experiment)
  - [Comparative Charts and Tables](#comparative-charts-and-tables)
- [Notebooks for Colab and Kaggle](#notebooks-for-colab-and-kaggle)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

The aim of this mini-project is to demonstrate a DRL (Deep Reinforcement Learning) approach for cryptocurrency trading by leveraging the FinRL framework. This enhanced fork adapts the original project to run on free GPU resources, enabling rapid prototyping via Google Colab and Kaggle. The main notebook to execute the project is:

- **NNDL_miniProj_FinRL-Crypto_NNinan.ipynb**

The project focuses on three distinct hyperparameter optimization schemes:
- **CPCV (Cascading Portfolio Cross-Validation)**
- **KCV (K-Cross Validation)**
- **WF (Walk-Forward)**

Each method optimizes the configuration parameters using Optuna trials. The outputs from these scripts are analyzed to provide insights into model performance, overfitting control, and overall strategy robustness.

---

## Repository Structure

```
FinRL_Crypto/
├── config_main/                   # Configuration files for models and hyperparameter ranges
├── drl_agents/                    # DRL agent implementations (adapted for fast execution)
├── notebooks/
│   └── NNDL_miniProj_FinRL-Crypto_NNinan.ipynb  # Main execution notebook
├── plots_visualisations/          # Folder containing generated plots and visualizations
├── presentation (1).pdf           # Prospective Summary PDF explaining the underlying research paper
├── 1_optimize_cpcv.py             # CPCV optimization script (with additional logging)
├── 1_optimize_kcv.py              # KCV optimization script (with additional logging)
├── 1_optimize_wf.py               # WF optimization script (with additional logging)
└── README.md                      # This README file
```

---

## Installation & Setup

1. **Clone the Repo:**

   ```bash
   git clone https://github.com/nidhinninan/FinRL_Crypto.git
   cd FinRL_Crypto
   ```

2. **Set Up the Environment:**
   
   Create and activate your Python virtual environment. Then install the requirements:

   ```bash
   pip install -r requirements.txt
   ```
   
3. **Google Colab / Kaggle Notebook:**

   This fork has been adapted for free GPU notebook environments. Open the provided notebooks in Google Colab or Kaggle, and ensure that the runtime is set to GPU to take full advantage of the resource.

---

## Experiment Details

### Hyperparameter Optimization

Each optimization script uses Optuna to search over a range of hyperparameters. The main parameters in play include:
- `learning_rate`
- `batch_size`
- `gamma`
- `net_dimension`
- `target_step`
- Time-gap settings
- Normalization factors for various signals (cash, stocks, tech, reward, action)

The core modifications include adjustments in the parameter ranges and the reduction of training samples and candle data for faster execution.

### Cross-Validation Methods

The experiments are structured around three cross-validation methods to assess model generalizability:

- **CPCV (Cascading Portfolio Cross-Validation):**  
  Designed to minimize overfitting by segmenting the data into distinct portfolios and testing across multiple groups.
  
- **KCV (K-Cross Validation):**  
  A more traditional partitioning technique that creates multiple folds to validate the robustness of the trained model.
  
- **WF (Walk-Forward):**  
  Evaluates how the model performs on a sequentially changing dataset, mimicking real-world trading where market conditions vary over time.

Each method is set up with:
- **Timeframe:** 5 minutes data intervals
- **Train Samples:** 10,000 data points
- **Validation Samples:** 2,500 data points
- **Dates:** Specific training and validation periods (e.g., 2022-03-17 to 2022-04-29)
- **Trials:** 7 optuna trials per script
- **Splitting Strategy:** Defined number of splits and groups
- **Hyperparameter Search Space (Optuna):** The following ranges were explored during the 7 trials:
    ```python
    sampled_erl_params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1.5e-2, 7.5e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1280]),
        "gamma": trial.suggest_categorical("gamma", [0.85, 0.99]),
        "net_dimension": trial.suggest_categorical("net_dimension", [2**9, 2**10]), # i.e., [512, 1024]
        "target_step": trial.suggest_categorical("target_step",
                                               [average_episode_step_min,
                                                round(1.5 * average_episode_step_min),
                                                2 * average_episode_step_min]),
        "eval_time_gap": trial.suggest_categorical("eval_time_gap", [60]), # Fixed value
        "break_step": trial.suggest_categorical("break_step", [3e4, 4.5e4]) # i.e., [30000, 45000]
    }
    ```
    *Note: `average_episode_step_min` depends on the specific validation split length.*

---

## Results and Analysis

### Quality of Experiment

Due to the reduced dataset (limited tickers, num_of_candles, and fewer Optuna trials), there is a noticeable compromise in the quality of the trained models. However, these settings allow for:
- Rapid iteration and prototyping.
- Reasonable statistical insights despite the constrained environment.
- The ability to quickly test and visualize changes to hyperparameters.

Even though the absolute performance metrics are lower than what the original paper suggests, the experiment provides valuable insights into:
- Sensitivity of DRL agents to hyperparameter variations.
- Comparative performance across different cross-validation methodologies.

### Comparative Charts and Tables

The following table summarizes the best trial values from the three different optimization approaches based on the outputs:

| Method | Best Trial Value  | Key Hyperparameter Settings                                      | Observations                                                  |
|--------|-------------------|------------------------------------------------------------------|---------------------------------------------------------------|
| **CPCV**  | -0.2582 (Trial 2) | `learning_rate=0.015`, `batch_size=512`, `net_dimension=512`, `target_step=12500` | Consistent performance improvement after lowering batch size. |
| **KCV**   | -0.01264 (Trial 1) | `learning_rate=0.015`, `batch_size=3080`, `net_dimension=512`, `target_step=18750` | Lowest loss value achieved; suggests better generalizability in K-fold setup.  |
| **WF**    | -0.10586 (Trial 2) | `learning_rate=0.015`, `batch_size=512`, `net_dimension=512`, `target_step=12500` | Walk-Forward validation shows relatively higher loss; may indicate model drift over time. |

> **Visualizations:**  
> - Please refer to the **plots_visualisations** folder for detailed plots such as learning curves (e.g., `plot_learning_curve_cpcv.jpg`, `plot_learning_curve_kcv.jpg`, and `plot_learning_curve_wf.jpg`).
> - The **presentation (1).pdf** in the master folder contains a comprehensive summary of the research paper and the detailed experiment rationale and results.  
> - Comparative charts (e.g., bar charts or line graphs comparing loss values and convergence patterns) have been generated to visually inspect the differences between CPCV, KCV, and WF methodologies.

#### Example Comparative Chart (Conceptual)

Below is a conceptual example of how one might display the comparative performance graphically:

```markdown
| Cross-Validation Method | Best Loss Value | Convergence Speed | Stability of Results |
|-------------------------|-----------------|-------------------|----------------------|
| CPCV                    | -0.2582         | Moderate          | Consistent across trials |
| KCV                     | -0.01264        | Fast              | Very stable results  |
| WF                      | -0.10586        | Slow              | Higher variance      |
```

*Note: Actual charts are available in the repository under the plots_visualisations folder.*

The discrepancy is stark and directly attributable to the **necessary reduction in computational resources and data** for this adapted version. The original paper show that the cpcv method yielded better results in manging overfitting isuues and there is high likelihold that the reason we got kcv as a better results is due to the models overfitting on the given test data and the paper also suggesting that we might need atleast about 50 optuna trials to get a 95% certaining result that the trained models isn't overfitted, which for our training was a meagre 7 trials. In other sample trials that were traing we sometime got better results for cpcv and sometimes of kcv. The discussed results here is the best of the all the trials that were run. The results here **do refute the findings of the original paper** but rather illustrate the challenge of achieving good performance with DRL in finance without sufficient data and computational budget for training and hyperparameter tuning and the discussing issue f crypto based model overfitting.

---

## Notebooks for Colab and Kaggle

The project is optimized for free GPU environments:
- **Google Colab:**  
  Simply upload the notebook `NNDL_miniProj_FinRL-Crypto_NNinan.ipynb` to Colab and run cell-by-cell.
  
- **Kaggle Notebook:**  
  Use the integrated Kaggle Notebook option to run the project. Ensure you select the free GPU runtime to leverage acceleration.

For both platforms, all dependencies are pre-installed (as specified in the requirements), and the code is tweaked to allow smooth execution within the limited runtime constraints.

---

## Future Work

- **Scaling Up:**  
  With access to paid GPU resources, re-run the experiments with the full dataset, higher number of trials, and extended training durations to potentially achieve performance metrics closer to those reported in the original paper.

- **Extended Hyperparameter Tuning:**  
  Broader parameter ranges and adaptive optimization strategies (e.g., Bayesian Optimization) could be explored to further refine the model performance.

- **Additional Validation Methods:**  
  Integrating robustness checks and further statistical tests to better control for overfitting and ensure model generalizability in different market conditions.

- **Real-World Deployment:**  
  Transition from the simulation to live market deployment, leveraging adaptive decision-making in real trading scenarios.

---

## References

- **Primary Literature:**  
  - *FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance*  
    ([ArXiv Link](https://arxiv.org/pdf/2111.09395))
  
  - *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting*
    ([ArXiv Link](https://arxiv.org/pdf/2209.05559))
  
- **Documentation and Resources:**  
  - [Official FinRL Documentation](https://finrl.readthedocs.io/)
  - [AI4Finance-Foundation/FinRL_Crypto GitHub Repository](https://github.com/AI4Finance-Foundation/FinRL_Crypto)

---

By exploring this repository, users can gain insights into the practical challenges and adaptations required for DRL-based trading in environments with limited computational resources. The comprehensive evaluation through various cross-validation methods provides a robust framework for further research and development in quantitative finance.

*Happy Trading and Experimenting!*

---
