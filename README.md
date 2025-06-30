# IOAI TST Kazakhstan - Day 2: Player Clustering Solution

This repository contains the solution for the second day's problem of the IOAI TST (Team Selection Test) in Kazakhstan. The objective was to cluster football (soccer) players based on their in-game attributes.

**Kaggle Competition Link:** [https://www.kaggle.com/competitions/tst-day-2](https://www.kaggle.com/competitions/tst-day-2)

## Table of Contents

  - [Problem Description](https://www.google.com/search?q=%23problem-description)
  - [Evaluation Metric](https://www.google.com/search?q=%23evaluation-metric)
  - [Solution Overview](https://www.google.com/search?q=%23solution-overview)
  - [Code Description](https://www.google.com/search?q=%23code-description)
  - [Results](https://www.google.com/search?q=%23results)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Usage](https://www.google.com/search?q=%23usage)

## Problem Description

The task was to group football players into clusters based on their various in-game statistics (e.g., passing, shooting, defense). A key challenge was that players could have multiple positions (e.g., {CM, CAM}), and the clustering needed to reflect these overlapping positional roles.

## Evaluation Metric

The clustering performance was evaluated using the **B-Cubed F1 score for multi-positional clustering**. This metric assesses how well players with overlapping positions are grouped together.

Our solution achieved a score of **0.534** on this metric.

## Solution Overview

The approach taken involves several steps:

1.  **Data Loading:** Reading the player data and the sample submission file.
2.  **Feature Engineering (Meta-features):** Creating aggregated "meta-features" from raw player statistics to represent broader skill sets (e.g., `attacking_skill`, `passing_ability`).
3.  **Goalkeeper Identification:** Separately identifying goalkeepers, as their skill sets are distinctly different from outfield players.
4.  **Data Splitting:** Dividing the dataset into goalkeepers and outfield players.
5.  **Preprocessing:** Applying imputation for missing values and standardization to scale features.
6.  **Optimal Cluster Determination:** Using the Silhouette Score to find an optimal number of clusters for outfield players.
7.  **Clustering:** Applying Gaussian Mixture Models (GMM) for clustering both outfield players and goalkeepers separately.
8.  **Cluster ID Consolidation:** Assigning unique cluster IDs to both groups.
9.  **Submission Generation:** Merging the clustered data with the sample submission format.

## Code Description

The provided Python script `batyr-yerdenov-2.2.ipynb` (intended to be run in a Jupyter/Kaggle notebook environment) implements the following:

  * **Import Libraries:** Essential libraries like `pandas`, `numpy`, `sklearn.preprocessing`, `sklearn.impute`, `sklearn.mixture`, `sklearn.metrics`.

  * **Data Loading:**

    ```python
    df = pd.read_csv("/kaggle/input/tst-day-2/train.csv")
    sample = pd.read_csv("/kaggle/input/tst-day-2/sample_submission.csv")
    ```

  * **Meta-feature Engineering:** New features are created by averaging relevant raw attributes. Examples include `attacking_skill`, `passing_ability`, `dribble_mobility`, `pace`, `defense_skill`, `physicality`, `set_piece_specialist`, `goalkeeper_score`, `composure_score`, `offensive_support`, `attack_support`, and `defending_positioning`.

  * **Goalkeeper Identification (`is_gk`):** A boolean column `is_gk` is created. A player is classified as a goalkeeper if all their specific goalkeeper attributes (`gk_diving`, `gk_handling`, `gk_kicking`, `gk_positioning`, `gk_reflexes`) are above a threshold of 40.

  * **Data Splitting:** The DataFrame `df` is split into `gk_df` (goalkeepers) and `field_df` (outfield players).

  * **Feature Selection for Clustering:**

      * `features`: A list of meta-features used for clustering outfield players.
      * `gk_features`: A list containing `goalkeeper_score` for clustering goalkeepers.

  * **Preprocessing Function (`preprocess`):**

    ```python
    def preprocess(X):
        X = SimpleImputer(strategy="mean").fit_transform(X) # Impute missing values with mean
        X = StandardScaler().fit_transform(X)             # Scale features to zero mean and unit variance
        return X
    ```

  * **Optimal Cluster Determination for Outfield Players:** A loop iterates from 5 to 14 clusters, fitting a `GaussianMixture` model and calculating the `silhouette_score`. The number of clusters yielding the highest silhouette score is chosen as `best_k` (found to be 6 in this run).

  * **Clustering with GMM:**

      * `gmm_field`: Gaussian Mixture Model fitted to `X_field` with `best_k` components.
      * `gmm_gk`: Gaussian Mixture Model fitted to `X_gk` with 1 component (assuming goalkeepers form a single, distinct cluster).

  * **Assigning Cluster Labels:** Cluster labels are predicted for both groups. Goalkeeper cluster IDs are offset by `best_k` to ensure unique IDs across the entire dataset.

  * **Combining Results:** The clustered `field_df` and `gk_df` are concatenated and sorted by player `id`.

  * **Submission Generation:** The final `submission.csv` file is created by merging the player `id` and their assigned `cluster` with the `sample_submission.csv`.

## Results

The solution achieved a **B-Cubed F1 score of 0.534**.

## Dependencies

  * `pandas`
  * `numpy`
  * `scikit-learn` (for `StandardScaler`, `SimpleImputer`, `GaussianMixture`, `silhouette_score`)

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1.  **Download the data:** Obtain `train.csv` and `sample_submission.csv` from the competition page ([https://www.kaggle.com/competitions/tst-day-2](https://www.kaggle.com/competitions/tst-day-2)) and place them in the specified path (`/kaggle/input/tst-day-2/`). If running locally, adjust the paths accordingly.
2.  **Run the Jupyter Notebook:** Open and run the `batyr-yerdenov-2.2.ipynb` notebook.
3.  **Generate Submission:** The script will automatically generate a `submission.csv` file in the same directory where the notebook is executed. This file will contain player IDs and their predicted cluster assignments.

-----
