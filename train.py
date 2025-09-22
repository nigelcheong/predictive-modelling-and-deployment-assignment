import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ===========================
# LOAD DATA
# ===========================
df_clean = pd.read_csv("data/engineered.csv")

# ===========================
# REGRESSION: FEATURE SETUP
# ===========================
reg_features = ['cost', 'lat', 'lng', 'votes', 'cost_2', 'rating_text_encoded', 'cuisine_diversity']
X_reg = df_clean[reg_features]
y_reg = df_clean['rating_number']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ===========================
# REGRESSION: SCALING
# ===========================
scaler = StandardScaler()
scaler.fit(X_train_reg)
X_train_reg_scaled = scaler.transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# ===========================
# REGRESSION: TRAIN MODEL
# ===========================
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg_scaled, y_train_reg)

# ===========================
# REGRESSION: PREDICTIONS
# ===========================
y_pred_lr = lin_reg.predict(X_test_reg_scaled)

# ===========================
# REGRESSION: EVALUATION
# ===========================
mse_lr = mean_squared_error(y_test_reg, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_reg, y_pred_lr)

# ===========================
# REGRESSION: SAVE METRICS
# ===========================
results_reg = pd.DataFrame([{
    'Model': 'Linear Regression',
    'MSE': mse_lr,
    'RMSE': rmse_lr,
    'R²': r2_lr
}])

results_path = "results/regression_results.csv"
os.makedirs("results", exist_ok=True)

try:
    existing = pd.read_csv(results_path)
    results_reg = pd.concat([existing, results_reg], ignore_index=True)
except FileNotFoundError:
    pass

results_reg.to_csv(results_path, index=False)

# ===========================
# REGRESSION: SAVE PREDICTIONS
# ===========================
preds_df = pd.DataFrame({
    'Actual': y_test_reg,
    'Predicted': y_pred_lr
})
preds_path = "results/linear_regression_predictions.csv"
preds_df.to_csv(preds_path, index=False)

print("Regression metrics and predictions saved.")

# ===========================
# CLASSIFICATION: FEATURE SETUP
# ===========================
clf_features = ['cost', 'lat', 'lng', 'votes', 'cost_2', 'cuisine_diversity']
X_clf = df_clean[clf_features]
y_clf = df_clean['rating_binary']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# ===========================
# CLASSIFICATION: SCALING
# ===========================
scaler_2 = StandardScaler()
scaler_2.fit(X_train_clf)
X_train_clf_scaled = scaler_2.transform(X_train_clf)
X_test_clf_scaled = scaler_2.transform(X_test_clf)

# ===========================
# CLASSIFICATION: TRAIN MODEL
# ===========================
svc = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42)
svc.fit(X_train_clf_scaled, y_train_clf)

# ===========================
# CLASSIFICATION: PREDICTIONS
# ===========================
y_pred_svc = svc.predict(X_test_clf_scaled)

# ===========================
# CLASSIFICATION: EVALUATION
# ===========================
acc_svc = accuracy_score(y_test_clf, y_pred_svc)
prec_svc = precision_score(y_test_clf, y_pred_svc)
rec_svc = recall_score(y_test_clf, y_pred_svc)
f1_svc = f1_score(y_test_clf, y_pred_svc)

# ===========================
# CLASSIFICATION: SAVE METRICS
# ===========================
results_clf = pd.DataFrame([{
    'Model': 'SVM Classifier (RBF)',
    'Accuracy': acc_svc,
    'Precision': prec_svc,
    'Recall': rec_svc,
    'F1': f1_svc
}])

results_path = "results/classifier_results.csv"
os.makedirs("results", exist_ok=True)

try:
    existing = pd.read_csv(results_path)
    results_clf = pd.concat([existing, results_clf], ignore_index=True)
except FileNotFoundError:
    pass

results_clf.to_csv(results_path, index=False)

# ===========================
# CLASSIFICATION: SAVE PREDICTIONS
# ===========================
preds_df = pd.DataFrame({
    'Actual': y_test_clf,
    'Predicted': y_pred_svc
})
preds_path = "results/svm_classifier_predictions.csv"
preds_df.to_csv(preds_path, index=False)

print("Classification metrics and predictions saved.")

os.makedirs("models", exist_ok=True)
with open("models/lin_reg_model.pkl", "wb") as f:
    pickle.dump(lin_reg, f)
with open("models/svc_model.pkl", "wb") as f:
    pickle.dump(svc, f)
    