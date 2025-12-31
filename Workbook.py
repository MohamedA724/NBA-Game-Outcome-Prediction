import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

pd.set_option("display.max_columns", 100)
sns.set()

# Load the games dataset (make sure games.csv is in the same folder as this notebook)

games = pd.read_csv("games.csv")
games.head()

games.info()
games[['PTS_home', 'PTS_away', 'HOME_TEAM_WINS']].describe()


# Keep only rows with non-null key stats and label, and drop duplicates

key_cols = [
    'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    'HOME_TEAM_WINS'
]

games_clean = games.dropna(subset=key_cols).drop_duplicates()
print("Original shape:", games.shape)
print("Cleaned shape:", games_clean.shape)


# Distribution of points for home vs away teams

plt.figure()
sns.histplot(games_clean['PTS_home'], bins=30, label='Home', stat='density', alpha=0.5)
sns.histplot(games_clean['PTS_away'], bins=30, label='Away', stat='density', alpha=0.5)
plt.legend()
plt.title("Distribution of points scored – Home vs Away")
plt.xlabel("Points")
plt.show()

# Home win rate overall
home_win_rate = games_clean['HOME_TEAM_WINS'].mean()
print(f"Overall home win rate: {home_win_rate:.3f}")

# 3. MODELING

# Create difference features: home stat minus away stat
games_fe = games_clean.copy()

games_fe['diff_pts']     = games_fe['PTS_home']     - games_fe['PTS_away']
games_fe['diff_fg_pct']  = games_fe['FG_PCT_home']  - games_fe['FG_PCT_away']
games_fe['diff_ft_pct']  = games_fe['FT_PCT_home']  - games_fe['FT_PCT_away']
games_fe['diff_fg3_pct'] = games_fe['FG3_PCT_home'] - games_fe['FG3_PCT_away']
games_fe['diff_ast']     = games_fe['AST_home']     - games_fe['AST_away']
games_fe['diff_reb']     = games_fe['REB_home']     - games_fe['REB_away']

diff_cols = [
    'diff_pts', 'diff_fg_pct', 'diff_ft_pct',
    'diff_fg3_pct', 'diff_ast', 'diff_reb'
]

X = games_fe[diff_cols]
y = games_fe['HOME_TEAM_WINS']

# Train / validation / test split: 60% / 20% / 20%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train.shape, X_val.shape, X_test.shape


def evaluate_predictions(name, y_true, y_pred, y_proba=None):
    print(f"\n=== {name} ===")
    print("Accuracy:",  accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:",    recall_score(y_true, y_pred))
    print("F1:",        f1_score(y_true, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_true, y_proba))

# Baseline 1: Always predict home wins (label = 1)
y_pred_home_always = np.ones_like(y_test)
evaluate_predictions("Baseline: Always Home Wins (test)", y_test, y_pred_home_always)

# Baseline 2: sanity check - predict home win if it scores more points
y_pred_by_points = (
    games_fe.loc[y_test.index, 'PTS_home'] >
    games_fe.loc[y_test.index, 'PTS_away']
).astype(int)

evaluate_predictions("Sanity: Home Scored More Points (test)", y_test, y_pred_by_points)


# Logistic Regression

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train, y_train)

y_val_proba_lr = log_reg.predict_proba(X_val)[:, 1]
y_val_pred_lr = (y_val_proba_lr >= 0.5).astype(int)

evaluate_predictions("Logistic Regression (val)", y_val, y_val_pred_lr, y_val_proba_lr)

# Logistic Regression

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train, y_train)

y_val_proba_lr = log_reg.predict_proba(X_val)[:, 1]
y_val_pred_lr = (y_val_proba_lr >= 0.5).astype(int)

evaluate_predictions("Logistic Regression (val)", y_val, y_val_pred_lr, y_val_proba_lr)

# Decision Tree

tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    random_state=42
)

tree.fit(X_train, y_train)

y_val_proba_tree = tree.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_proba_tree >= 0.5).astype(int)

evaluate_predictions("Decision Tree (val)", y_val, y_val_pred_tree, y_val_proba_tree)

# Random Forest

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_val_proba_rf = rf.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_proba_rf >= 0.5).astype(int)

evaluate_predictions("Random Forest (val)", y_val, y_val_pred_rf, y_val_proba_rf)

# Choose Random Forest as final model (change if another model performs better on val)

y_test_proba_rf = rf.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_proba_rf >= 0.5).astype(int)

evaluate_predictions("Random Forest (test)", y_test, y_test_pred_rf, y_test_proba_rf)
print("\nClassification report (Random Forest, test):")
print(classification_report(y_test, y_test_pred_rf))

cm = confusion_matrix(y_test, y_test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Random Forest – Confusion Matrix (Test)")
plt.show()

# ROC curves for all three models on the validation set

models_proba = {
    "LogReg": y_val_proba_lr,
    "DecisionTree": y_val_proba_tree,
    "RandomForest": y_val_proba_rf,
}

plt.figure()
for name, proba in models_proba.items():
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc = roc_auc_score(y_val, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Validation)")
plt.legend()
plt.show()

# Random Forest feature importance
importances = pd.Series(rf.feature_importances_, index=diff_cols).sort_values()

plt.figure()
importances.plot(kind="barh")
plt.title("Random Forest – Feature Importance")
plt.xlabel("Importance")
plt.show()

importances

# Logistic Regression coefficients
coef_lr = pd.Series(log_reg.coef_[0], index=diff_cols).sort_values()

plt.figure()
coef_lr.plot(kind="barh")
plt.title("Logistic Regression – Coefficients")
plt.xlabel("Coefficient")
plt.show()

coef_lr

