"""
Student Focus Behavior Classifier
Behavioral ML system to classify student study sessions as high-focus or low-focus
and identify the behavioral drivers behind each pattern.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("STUDENT FOCUS BEHAVIOR CLASSIFIER")
print("=" * 55)

# Generate inline if CSV not present
try:
    df = pd.read_csv("data/student_focus_behavior.csv")
except FileNotFoundError:
    import sys
    sys.path.insert(0, "data")
    from generate_dataset import generate_dataset
    df = generate_dataset()

print(f"\n[DATA] Shape: {df.shape}")
print(f"[DATA] High-focus sessions: {df['high_focus'].mean():.1%}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
# Drop derived score (it's the label source, not a real feature)
df_model = df.copy()

# Encode categoricals
le_noise = LabelEncoder()
le_time = LabelEncoder()
df_model["background_noise_level"] = le_noise.fit_transform(df_model["background_noise_level"])
df_model["time_of_day"] = le_time.fit_transform(df_model["time_of_day"])

X = df_model.drop(columns=["high_focus"])
y = df_model["high_focus"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

## Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42, C=1.0)
lr.fit(X_train_sc, y_train)

lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring="roc_auc")
print(f"\n[LR] Cross-val AUC: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")

## Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print(f"[RF] Cross-val AUC: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("EVALUATION — RANDOM FOREST (selected model)")
print("─" * 55)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf, target_names=["Low Focus", "High Focus"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

print("\n" + "─" * 55)
print("EVALUATION — LOGISTIC REGRESSION")
print("─" * 55)
y_pred_lr = lr.predict(X_test_sc)
y_proba_lr = lr.predict_proba(X_test_sc)[:, 1]
print(classification_report(y_test, y_pred_lr, target_names=["Low Focus", "High Focus"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────

## Feature Importance (RF)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Student Focus Behavior Classifier — Model Insights", fontsize=14, fontweight="bold")

# --- Plot 1: Feature Importances ---
colors = ["#2563EB" if imp > 0.08 else "#93C5FD" for imp in sorted_importances]
axes[0].barh(sorted_features[::-1], sorted_importances[::-1], color=colors[::-1])
axes[0].set_xlabel("Importance Score")
axes[0].set_title("Feature Importances (Random Forest)")
axes[0].axvline(x=0.08, color="red", linestyle="--", alpha=0.5, label="Threshold (0.08)")
axes[0].legend(fontsize=8)

# --- Plot 2: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Focus", "High Focus"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix (Random Forest)")

# --- Plot 3: ROC Curves ---
RocCurveDisplay.from_predictions(
    y_test,
    y_proba_rf,
    ax=axes[2],
    name=f"Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.2f})"
)

RocCurveDisplay.from_predictions(
    y_test,
    y_proba_lr,
    ax=axes[2],
    name=f"Logistic Reg (AUC={roc_auc_score(y_test, y_proba_lr):.2f})"
)
axes[2].set_title("ROC Curve Comparison")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)


plt.tight_layout()
plt.savefig("outputs/model_insights.png", dpi=150, bbox_inches="tight")
print("\n[SAVED] outputs/model_insights.png")

## Focus probability distribution
fig2, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_proba_rf[y_test == 0], bins=20, alpha=0.7, color="#EF4444", label="Low Focus", edgecolor="white")
ax.hist(y_proba_rf[y_test == 1], bins=20, alpha=0.7, color="#22C55E", label="High Focus", edgecolor="white")
ax.axvline(x=0.5, color="black", linestyle="--", label="Decision Boundary")
ax.set_xlabel("Predicted Focus Probability")
ax.set_ylabel("Count")
ax.set_title("Focus Probability Distribution by True Class")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/probability_distribution.png", dpi=150, bbox_inches="tight")
print("[SAVED] outputs/probability_distribution.png")

# ─────────────────────────────────────────────
# 6. INFERENCE DEMO
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("INFERENCE DEMO — Predict a new session")
print("─" * 55)

# Encode maps
noise_map = dict(zip(le_noise.classes_, le_noise.transform(le_noise.classes_)))
time_map = dict(zip(le_time.classes_, le_time.transform(le_time.classes_)))

sample_session = {
    "session_duration_min": 45,
    "break_frequency": 1,
    "phone_pickups": 3,
    "tab_switches": 5,
    "task_completion_rate": 0.82,
    "self_rated_focus": 7,
    "background_noise_level": noise_map["low"],
    "time_of_day": time_map["morning"],
    "days_since_last_break": 1,
    "caffeine_intake_cups": 1,
    "prior_sleep_hours": 7.5,
    "streak_days": 12
}

sample_df = pd.DataFrame([sample_session])
prob = rf.predict_proba(sample_df)[0][1]
label_str = "HIGH FOCUS ✓" if prob >= 0.5 else "LOW FOCUS ✗"
print(f"Session Prediction: {label_str}")
print(f"Focus Probability: {prob:.1%}")

print("\n[DONE] All outputs saved to /outputs/")
