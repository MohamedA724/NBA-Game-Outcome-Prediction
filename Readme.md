# NBA Home Team Win Prediction 🏀📈

Predict whether the **home team wins** an NBA game using historical game statistics. This project builds an end-to-end machine learning pipeline including **data cleaning**, **exploratory data analysis**, **feature engineering**, and **model training/evaluation**.

---

## Table of Contents
- Overview
- Dataset
- Approach
- Results & Visualizations
- How to Run
- Project Structure
- Tech Stack
- Future Improvements

---

## Overview
The goal of this project is to predict the binary outcome:

**HOME_TEAM_WINS**  
1 = Home team wins  
0 = Home team loses

Using team-level box score statistics for both home and away teams, multiple machine learning classifiers are trained and compared against simple baselines.

---

## Dataset
The project expects a file named:

**games.csv**

located in the project directory.

The dataset contains:
- Points scored
- Shooting percentages (FG%, FT%, 3P%)
- Assists
- Rebounds
- Game outcome label (`HOME_TEAM_WINS`)

Rows with missing values in key statistics are removed before training.

---

## Approach

### Feature Engineering
Rather than using raw team statistics, the model uses **difference features**:

`home_stat − away_stat`

Features include:
- Points difference
- Field goal percentage difference
- Free throw percentage difference
- Three-point percentage difference
- Assist difference
- Rebound difference

This highlights the relative advantage of the home team.

---

### Models
The following models are implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (final model)

---

### Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional analysis includes:
- Confusion matrices
- ROC curves
- Feature importance (Random Forest)
- Model coefficients (Logistic Regression)

---

## Results & Visualizations
The project generates visualizations such as:
- Home vs away score distributions
- Confusion matrix for the final model
- ROC curves comparing models
- Feature importance plots

---

## How to Run

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the project
Ensure `games.csv` is in the project directory, then run the notebook or script.

---

## Project Structure
```
.
├── games.csv
├── nba_prediction.ipynb
├── nba_prediction.py
└── README.md
```

---

## Tech Stack
- Python
- Pandas & NumPy
- scikit-learn
- Matplotlib & Seaborn

---

## Future Improvements
- Add additional team statistics and contextual features
- Perform hyperparameter tuning
- Use cross-validation
- Add model explainability tools (SHAP/LIME)
- Deploy as a web application

---

This project is intended for educational and portfolio purposes.
