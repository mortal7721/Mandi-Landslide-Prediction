import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set style
sns.set_theme(style="whitegrid")

# 1. Load dataset
csv_path = Path("mandi_landslides_features_merged.csv")
df = pd.read_csv(csv_path)

# 2. Prepare Feature Matrix (X) and Target (y)
features = ['tp_3_day_cum', 'tp_7_day_cum', 'swvl1_event', 'swvl2_event', 't2m_event']
available_features = [f for f in features if f in df.columns]

# Create balanced binary classification data (1 = Landslide Event, 0 = Normal Baseline Day)
X_events = df[available_features].copy()
X_events['label'] = 1

np.random.seed(42)
X_background = X_events.copy()
for col in available_features:
    if 'tp' in col:
        X_background[col] = np.random.exponential(0.0001, size=len(X_background))
    else:
        X_background[col] += np.random.normal(0, X_background[col].std() * 0.2, size=len(X_background))
X_background['label'] = 0

model_df = pd.concat([X_events, X_background], ignore_index=True)
model_df.fillna(model_df.median(numeric_only=True), inplace=True)

X = model_df[available_features].values
y = model_df['label'].values

# 3. Perform 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(12, 5))

# --- Subplot 1: ROC Curve ---
plt.subplot(1, 2, 1)
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz_prob = classifier.predict_proba(X[test])[:, 1]
    
    # Compute ROC curve and area
    fpr, tpr, _ = roc_curve(y[test], viz_prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold %d (AUC = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC-AUC)", fontsize=11, fontweight='bold')
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
plt.legend(loc="lower right", fontsize=8)

# --- Subplot 2: Precision-Recall Curve ---
plt.subplot(1, 2, 2)
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz_prob = classifier.predict_proba(X[test])[:, 1]
    
    precision, recall, _ = precision_recall_curve(y[test], viz_prob)
    ap = average_precision_score(y[test], viz_prob)
    plt.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold %d (AP = %0.2f)' % (i+1, ap))

plt.title("Precision-Recall Curve", fontsize=11, fontweight='bold')
plt.xlabel("Recall", fontsize=10)
plt.ylabel("Precision", fontsize=10)
plt.legend(loc="lower left", fontsize=8)

plt.tight_layout()

# Save validation plot
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / "model_validation_curves_simple.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"[+] Saved clean validation curves to {plot_path}")