from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)
CSV_PATH = BASE / "data.csv"
if not CSV_PATH.exists():
    found = sorted(BASE.glob("*.csv"))
    if not found:
        raise SystemExit("Put the dataset CSV next to analysis.py (named data.csv)")
    CSV_PATH = found[0]
df = pd.read_csv(CSV_PATH).drop(columns=["Unnamed: 0"])
df["visited"] = (df["visits"] > 0).astype(int)
features = [c for c in df.columns if c not in ("visits", "visited")]
categorical = ["gender", "private", "freepoor", "freerepat", "nchronic", "lchronic"]
numeric = [c for c in features if c not in categorical]
y = df["visited"]
X = df[features]

plt.figure(figsize=(6, 4))
df["visits"].value_counts().sort_index().plot(kind="bar", color="#3678a8")
plt.title("Doctor Visit Counts")
plt.xlabel("Number of visits")
plt.ylabel("People")
plt.tight_layout()
plt.savefig(OUT / "visits_dist.png", dpi=120)
plt.close()

groups = ["gender", "lchronic", "nchronic", "freerepat", "freepoor", "private"]
fig, ax = plt.subplots(figsize=(6, 4))
levels = [sorted(df[col].dropna().unique()) for col in groups]
width = 0.8 / max(map(len, levels))
for i, (col, values) in enumerate(zip(groups, levels)):
    means = [df.loc[df[col] == value, "visits"].mean() for value in values]
    offsets = (np.arange(len(values)) - (len(values) - 1) / 2) * width
    positions = np.arange(len(groups))[i] + offsets
    ax.bar(positions, means, width, color="#3678a8")
    for position, value, category in zip(positions, means, values):
        ax.text(position, value, str(category), ha="center", va="bottom", fontsize=6, rotation=45)
ax.set_xticks(range(len(groups)), groups)
ax.set_title("Mean Visits by Group")
ax.set_ylabel("Mean visits")
plt.tight_layout()
plt.savefig(OUT / "mean_visits_by_group.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(6, 4))
for ax, col in zip(axes, ["illness", "reduced"]):
    means = df.groupby(col, observed=False)["visits"].mean().sort_index()
    ax.plot(means.index, means.values, marker="o", color="#3678a8")
    ax.set_title(f"Mean Visits vs {col.title()}")
    ax.set_xlabel(col.title())
    ax.set_ylabel("Mean visits")
plt.tight_layout()
plt.savefig(OUT / "illness_reduced.png", dpi=120)
plt.close()

encoded = pd.get_dummies(X, columns=categorical, drop_first=True)
corr = encoded.corrwith(df["visits"]).dropna().sort_values()
plt.figure(figsize=(6, 4))
corr.plot(kind="barh", color="#3678a8")
plt.title("Feature Correlation with Visits")
plt.xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig(OUT / "corr.png", dpi=120)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def make_preprocessor(scale=False):
    numeric_transform = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("numeric", numeric_transform, numeric), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)])

models = {
    "LogisticRegression": Pipeline([("preprocessor", make_preprocessor(True)), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
    "RandomForest": Pipeline([("preprocessor", make_preprocessor()), ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42))]),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    results[name] = (model, accuracy_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, prob), pred)
    print(f"{name}: accuracy={results[name][1]:.4f}, F1={results[name][2]:.4f}, ROC-AUC={results[name][3]:.4f}")

best_name = max(results, key=lambda name: results[name][3])
best_model, _, _, _, best_pred = results[best_name]
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(cm, cmap="Blues")
ax.set_title(f"Confusion Matrix: {best_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1], ["No", "Yes"])
ax.set_yticks([0, 1], ["No", "Yes"])
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT / "confusion.png", dpi=120)
plt.close()

rf = models["RandomForest"]
rf.fit(X, y)
importance = pd.Series(rf.named_steps["model"].feature_importances_, index=[n.split("__",1)[1] for n in rf.named_steps["preprocessor"].get_feature_names_out()]).sort_values()
plt.figure(figsize=(6, 4))
importance.plot(kind="barh", color="#3678a8")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT / "feat_importance.png", dpi=120)
plt.close()
best_model.fit(X, y)
joblib.dump(best_model, BASE / "model.pkl")
print(f"Saved model.pkl ({best_name}); charts in {OUT}")
