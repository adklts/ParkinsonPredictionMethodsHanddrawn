#!/usr/bin/env python
# coding: utf-8
"""Clean Python export from Spiralv1.ipynb.

Repeated import statements from the notebook were consolidated into one import section.
Notebook shell/magic commands were converted to comments.
"""

# === Consolidated imports ===
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import joblib
from scipy.signal import find_peaks, welch
from scipy.stats import (
    entropy,
    skew,
    kurtosis,
    ttest_ind,
    shapiro,
    pearsonr,
    spearmanr,
    mannwhitneyu,
    wasserstein_distance,
)
from scipy.fftpack import fft
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except ImportError:
    display = print


# %% [Code cell 1]
# Create the folder if it doesn't exist
output_folder = "boxplots"
os.makedirs(output_folder, exist_ok=True)


def extract_wave_features(y_coordinates):
    """Extract wave-based features from y-coordinates."""
    if len(y_coordinates) < 2:
        return [np.nan] * 9

    # Normalize
    signal = (y_coordinates - np.min(y_coordinates)) / (np.max(y_coordinates) - np.min(y_coordinates) + 1e-6)
    signal_diff = np.diff(signal)

    # Peaks and troughs
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)

    # Zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal - np.mean(signal))))[0]

    # Energy
    energy = np.sum(signal ** 2)

    # Entropy (histogram based)
    hist, _ = np.histogram(signal, bins=10, density=True)
    signal_entropy = entropy(hist + 1e-6)

    # Skewness & Kurtosis
    sk = skew(signal)
    kurt_val = kurtosis(signal)

    # FFT and Dominant Frequency
    spectrum = np.abs(fft(signal))
    dom_freq = np.argmax(spectrum[1:]) + 1

    return [
        len(peaks),
        len(troughs),
        len(zero_crossings),
        energy,
        signal_entropy,
        sk,
        kurt_val,
        dom_freq
    ]

def extract_features(image_path):
    """Extract standard and extra wave features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    y_coordinates = np.argwhere(edges > 0)[:, 0]

    if len(y_coordinates) < 2:
        return [np.nan] * 12

    # Core Features
    amplitude_diff = np.diff(y_coordinates)
    smoothness = np.var(amplitude_diff)
    avg_amplitude = np.mean(amplitude_diff)
    irregularities = len(np.where(np.abs(amplitude_diff) > avg_amplitude * 1.5)[0])

    # Extra Features
    extra = extract_wave_features(y_coordinates)

    return [smoothness, avg_amplitude, irregularities] + extra

def process_folder(folder_path, label):
    """Processes all images in a folder and extracts features."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            data.append({
                "Image": filename,
                "Label": label,
                "Smoothness": features[0],
                "Average Amplitude": features[1],
                "Irregularities": features[2],
                "Peak Count": features[3],
                "Trough Count": features[4],
                "Zero Crossings": features[5],
                "Wave Energy": features[6],
                "Signal Entropy": features[7],
                "Skewness": features[8],
                "Kurtosis": features[9],
                "Dominant Frequency": features[10]
            })
    return data

# === Set Folder Paths ===
#healthy_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\Paper\PhDOntologiesSpiral(Waves)\spiral\training\Healthy'
#parkinsonian_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\Paper\PhDOntologiesSpiral(Waves)\spiral\training\Parkinson'
healthy_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\Paper\PhDOntologiesSpiral(Waves)\spiral\testing\Healthy'
parkinsonian_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\Paper\PhDOntologiesSpiral(Waves)\spiral\testing\Parkinson'

# === Process Images ===
healthy_data = process_folder(healthy_folder, "Healthy")
parkinsonian_data = process_folder(parkinsonian_folder, "Parkinsonian")
all_data = healthy_data + parkinsonian_data
df = pd.DataFrame(all_data)

# === Save and Preview ===
df.to_csv("extracted_featuresnewtest.csv", index=False)
print("Saved to extracted_featurestesting.csv")
print(df.head())

# === Summary Stats ===
print("\n=== Summary Statistics by Label ===")
print(df.groupby("Label").mean(numeric_only=True))

# === Boxplots per Feature ===
features_to_plot = [col for col in df.columns if col not in ['Image', 'Label']]
for feature in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Label', y=feature, data=df)
    plt.title(f"Boxplot of {feature}")
    plt.tight_layout()

    # Save the figure
    filename = f"{feature}_boxplot.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)

    plt.close()  # Close the figure to free memory


# %% [Code cell 2]
# === Descriptive Statistics for Each Feature by Label ===
print("\n=== Full Descriptive Statistics by Label ===")

# Get grouped descriptive stats
grouped_stats = df.groupby("Label")[features_to_plot].describe()

# Loop through features and print cleanly
for feature in features_to_plot:
    print(f"\n📊 === {feature} ===")
    display(grouped_stats[feature])  # works in Jupyter / shows nicely in console


# %% [Code cell 3]
# Build one big text block
grouped_stats = df.groupby("Label")[features_to_plot].describe()

lines = []
lines.append("=== Full Descriptive Statistics by Label ===\n")

for feature in features_to_plot:
    lines.append(f"📊 === {feature} ===")
    # Convert table to a nice text layout
    lines.append(grouped_stats[feature].round(6).to_string())
    lines.append("")  # blank line between features

text_block = "\n".join(lines)

# ---- Render text into ONE image ----
# Adjust figsize based on text length (simple heuristic)
n_lines = text_block.count("\n") + 1
fig_height = max(10, n_lines * 0.22)   # tweak if needed

fig = plt.figure(figsize=(16, fig_height), dpi=200)
plt.axis("off")

plt.text(
    0.01, 0.99, text_block,
    va="top", ha="left",
    family="monospace", fontsize=10
)

plt.tight_layout()
plt.savefig("descriptive_stats_ALL.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: descriptive_stats_ALL.png")


# %% [Code cell 4]
#Enough good 18/4/2025 my last use

# -----------------------------------------------------
# ⚙️ CONFIGURATION
# -----------------------------------------------------
USE_TVAE = False  # Set to True to try TVAE instead of CTGAN
SYNTH_PER_CLASS = 500
EPOCHS = 12000
SYNTH_TOTAL = 200000
SAVE_TP_FILES = True
RUN_EVALUATION = True

# -----------------------------------------------------
# 🧪 Load and preprocess real data
# -----------------------------------------------------
df_real = pd.read_csv("extracted_featuresnew.csv")
if "Image" in df_real.columns:
    df_real = df_real.drop(columns=["Image"])

df_real["Label"] = df_real["Label"].astype(str)

# -----------------------------------------------------
# 🧠 Metadata detection
# -----------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_real)

# -----------------------------------------------------
# 🚀 Train model
# -----------------------------------------------------
if USE_TVAE:
    from sdv.single_table import TVAESynthesizer
    model = TVAESynthesizer(metadata, epochs=EPOCHS)
    print("🚀 Training TVAE...")
else:
    model = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS,
        batch_size=100,
        generator_lr=1e-4,
        discriminator_lr=1e-4
    )
    print("🚀 Training CTGAN...")

model.fit(df_real)
print("✅ Training complete.")

# -----------------------------------------------------
# 🎲 Sample synthetic data
# -----------------------------------------------------
print(f"📊 Sampling {SYNTH_TOTAL} synthetic rows...")
df_synth_all = model.sample(SYNTH_TOTAL)

# Filter to exactly N per class
synth_healthy = df_synth_all[df_synth_all["Label"] == "Healthy"].head(SYNTH_PER_CLASS)
synth_parkinsonian = df_synth_all[df_synth_all["Label"] == "Parkinsonian"].head(SYNTH_PER_CLASS)

df_synth = pd.concat([synth_healthy, synth_parkinsonian], ignore_index=True)

# Add synthetic image IDs
df_synth["Image"] = [
    f"Healthysynth{i+1}" if label == "Healthy" else f"Parkinsoniansynth{i+1}"
    for i, label in enumerate(df_synth["Label"])
]

output_file = "ctgan_synthetic_500_per_class.csv"
df_synth.to_csv(output_file, index=False)
print(f"✅ Saved {len(df_synth)} synthetic samples to → {output_file}")

# -----------------------------------------------------
# 🧠 Classifier Evaluation
# -----------------------------------------------------
if RUN_EVALUATION:
    print("\n🔍 Running classifier-based diagnostic evaluation...")

    X_real = df_real.drop(columns=["Label"])
    y_real = df_real["Label"]

    X_fake = df_synth.drop(columns=["Label", "Image"])
    y_fake = df_synth["Label"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_real, y_real)

    y_pred = clf.predict(X_fake)

    print("\n📊 Classification Report (Trained on Real → Tested on Synthetic):")
    print(classification_report(y_fake, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_fake, y_pred, labels=["Healthy", "Parkinsonian"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Parkinsonian"],
                yticklabels=["Healthy", "Parkinsonian"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: Trained on Real → Tested on Synthetic")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # ✅ Save only True Positive samples
    # -----------------------------------------------------
    print("\n📥 Extracting True Positive synthetic samples...")
    true_positive_mask = y_pred == y_fake.values
    df_true_positives = df_synth[true_positive_mask]
    print(f"✔ Found {len(df_true_positives)} True Positives")

    if SAVE_TP_FILES:
        # Save full TP set
        #df_true_positives.to_csv("ctgan_true_positives_all500v1.csv", index=False)
        #df_true_positives.to_csv("ctgan_true_positives_all500v1d.csv", index=False)
        # Save per-label TPs
        for label in ["Healthy", "Parkinsonian"]:
            df_tp_label = df_true_positives[df_true_positives["Label"] == label]
            out = f"ctgan_truepositives_{label.lower()}.csv"
            df_tp_label.to_csv(out, index=False)
            print(f"💾 Saved {len(df_tp_label)} True Positives for {label} → {out}")


# %% [Code cell 5]
# Load the two CSV files
df1 = pd.read_csv('ctgan_true_positives_all500v1.csv')
df2 = pd.read_csv('ctgan_true_positives_all500v1d.csv')

# Concatenate the dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('ctgan_true_positives_all1000.csv', index=False)

print("Files merged successfully into ctgan_true_positives_all1000.csv")


# %% [Code cell 6]
# === Create output folder ===
os.makedirs("comparision", exist_ok=True)

# === Load real and synthetic ===
df_real = pd.read_csv("extracted_featuresnew.csv")
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")

# === Match columns ===
feature_cols = [col for col in df_real.columns if col not in ["Image", "Label"]]

# === Ensure Label is same format ===
df_real["Origin"] = "Real"
df_synth["Origin"] = "Synthetic"
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# === 1. Boxplots per feature ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Origin", y=col, data=df_combined)
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"comparision/boxplot_{col}.png", dpi=300)
    plt.close()

# === 2. Overlaid histograms ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_real[col], color="blue", label="Real", kde=True, stat="density")
    sns.histplot(df_synth[col], color="orange", label="Synthetic", kde=True, stat="density")
    plt.title(f"Histogram: {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparision/histogram_{col}.png", dpi=300)
    plt.close()

# === 3. T-test comparison per feature ===
for col in feature_cols:
    if df_real[col].nunique() < 2 or df_synth[col].nunique() < 2:
        print(f"{col:25}: Skipped (not enough variation)")
        continue

    try:
        t_stat, p_value = ttest_ind(df_real[col], df_synth[col], equal_var=False)
        result = "Same Distribution ✅" if p_value > 0.05 else "Different ⚠️"
        print(f"{col:25}: p = {p_value:.4f} → {result}")
    except Exception as e:
        print(f"{col:25}: Error — {e}")


# %% [Code cell 7]
# === Load real and synthetic ===
df_real = pd.read_csv("extracted_featuresnew.csv")



df_synth = pd.read_csv("ctgan_true_positives_all500v1d.csv")

# === Match columns ===
feature_cols = [col for col in df_real.columns if col not in ["Image", "Label"]]

# === Ensure Label is same format ===
df_real["Origin"] = "Real"
df_synth["Origin"] = "Synthetic"
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# === 1. Boxplots per feature ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Origin", y=col, data=df_combined)
    plt.title(f"📦 Boxplot: {col}")
    plt.tight_layout()
    plt.show()

# === 2. Overlaid histograms ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_real[col], color="blue", label="Real", kde=True, stat="density")
    sns.histplot(df_synth[col], color="orange", label="Synthetic", kde=True, stat="density")
    plt.title(f"📈 Histogram: {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === 3. T-test comparison per feature ===
for col in feature_cols:
    if df_real[col].nunique() < 2 or df_synth[col].nunique() < 2:
        print(f"{col:25}: Skipped (not enough variation)")
        continue

    try:
        t_stat, p_value = ttest_ind(df_real[col], df_synth[col], equal_var=False)
        result = "Same Distribution ✅" if p_value > 0.05 else "Different ⚠️"
        print(f"{col:25}: p = {p_value:.4f} → {result}")
    except Exception as e:
        print(f"{col:25}: Error — {e}")


# %% [Code cell 9]
# Notebook shell command removed: !pip install shap


# %% [Code cell 10]
# === 1. Load data ===
#df = pd.read_csv("ctgan_true_positives_all1000.csv")


# Load the two CSV files
df = pd.read_csv('ctgan_true_positives_all1000.csv')
#df2 = pd.read_csv('ctgan_true_positives_all500v2.csv')




# === 2. Target variable (convert to 0/1) ===
df["target"] = df["Label"].map({"Healthy": 0, "Parkinsonian": 1})

# === 3. Define features (drop non-numeric or ID columns) ===
X = df.drop(columns=["Label", "Image", "target"])
y = df["target"]

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === 5. Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === 6. Evaluate ===
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 7. Feature importance (optional) ===
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind="barh", figsize=(8, 5), title="Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


green_features = ["Smoothness", "Kurtosis", "Signal Entropy", "Wave Energy", "Irregularities", "Skewness"]

# === 8. SHAP Explanation ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle 3D output for binary classifier
if isinstance(shap_values, list):
    shap_arr = shap_values[1]  # use SHAP values for class 1 (Parkinsonian)
else:
    shap_arr = shap_values

# If 3D, reduce to 2D using the second class (axis 2 index 1)
if shap_arr.ndim == 3:
    shap_arr = shap_arr[:, :, 1]

# Plot only if shape matches
if shap_arr.shape == X_test.shape:
    print("\n📈 SHAP Summary Plots:")
    shap.summary_plot(shap_arr, X_test, plot_type="bar")
    shap.summary_plot(shap_arr, X_test)
else:
    print(f"❌ SHAP shape mismatch. SHAP: {shap_arr.shape}, X_test: {X_test.shape}")



# === Normality Tests (Shapiro-Wilk) ===
print("\n📊 Shapiro-Wilk Normality Test:")
normality = {}
for col in green_features:
    stat, p = shapiro(df[col])
    normality[col] = "Normal" if p > 0.05 else "Not Normal"
    print(f"{col:<20}: p = {p:.4f} → {normality[col]}")

# === Correlation Tests ===
print("\n🔗 Correlation with Label (0=Healthy, 1=Parkinsonian):")
for col in green_features:
    if normality[col] == "Normal":
        corr, pval = pearsonr(df[col], y)
        test = "Pearson"
    else:
        corr, pval = spearmanr(df[col], y)
        test = "Spearman"
    print(f"{col:<20}: {test:<8} r = {corr:.3f}, p = {pval:.4f}")


# %% [Code cell 11]
# === 8. Save the trained model ===
joblib.dump(model, "random_forest_ctgan_model.pkl")
print("💾 Model saved as random_forest_ctgan_model.pkl")

# ---------------------------------------------------
# 🧪 Apply model to extracted real data
# ---------------------------------------------------

# === Load real data ===
df_real = pd.read_csv("extracted_featuresnewtest.csv")

# Convert label to 0/1 target
df_real["target"] = df_real["Label"].map({"Healthy": 0, "Parkinsonian": 1})

# Drop non-feature columns
X_real = df_real.drop(columns=["Label", "Image", "target"], errors="ignore")
y_real = df_real["target"]

# === Load model and predict ===
model = joblib.load("random_forest_ctgan_model.pkl")
y_real_pred = model.predict(X_real)

# === Evaluate on real data ===
print("\n🔍 Evaluation on Real Data:")
print(classification_report(y_real, y_real_pred))
print("🧩 Confusion Matrix (Real Data):")
print(confusion_matrix(y_real, y_real_pred))
print(f"✅ Accuracy: {accuracy_score(y_real, y_real_pred):.4f}")


# %% [Code cell 12]
# ========================================
# DATASET 2 (SPIRALS) — MULTIPLE TSTR RUNS
# ========================================


# ========================================
# 1. LOAD DATA
# ========================================
# Synthetic data (CTGAN-generated for spirals, used for training)
synthetic_df = pd.read_csv("ctgan_true_positives_all1000.csv")  # ← ΑΛΛΑΞΕ αν το spiral synthetic έχει άλλο όνομα

# Real test data for SPIRALS
df_real = pd.read_csv("extracted_featuresnewtest.csv")

# Convert labels to 0/1
synthetic_df["target"] = synthetic_df["Label"].map({"Healthy": 0, "Parkinsonian": 1})
df_real["target"] = df_real["Label"].map({"Healthy": 0, "Parkinsonian": 1})

# Drop non-feature columns
X_synthetic = synthetic_df.drop(columns=["Label", "Image", "target"], errors="ignore")
y_synthetic = synthetic_df["target"]

X_real = df_real.drop(columns=["Label", "Image", "target"], errors="ignore")
y_real = df_real["target"]

# Ensure same column order
X_real = X_real[X_synthetic.columns]

print(f"Synthetic samples (training): {len(X_synthetic)}")
print(f"Real test samples:            {len(X_real)}")
print(f"Features used:                {list(X_synthetic.columns)}\n")

# ========================================
# 2. MULTIPLE TSTR RUNS
# ========================================
N_RUNS = 50

f1_macro_scores = []
f1_class0_scores = []
f1_class1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []

print("="*80)
print(f"DATASET 2 (SPIRALS) — TSTR EVALUATION — {N_RUNS} RUNS")
print("="*80)
print(f"{'Run':<6} {'Seed':<8} {'F1_macro':<12} {'F1_Healthy':<12} {'F1_PD':<12} {'Accuracy':<10}")
print("-"*80)

for i in range(N_RUNS):
    seed = i

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_synthetic, y_synthetic)

    y_pred = model.predict(X_real)

    f1_macro = f1_score(y_real, y_pred, average='macro')
    f1_per_class = f1_score(y_real, y_pred, average=None)
    precision = precision_score(y_real, y_pred, average='macro')
    recall = recall_score(y_real, y_pred, average='macro')
    accuracy = accuracy_score(y_real, y_pred)

    f1_macro_scores.append(f1_macro)
    f1_class0_scores.append(f1_per_class[0])
    f1_class1_scores.append(f1_per_class[1])
    precision_scores.append(precision)
    recall_scores.append(recall)
    accuracy_scores.append(accuracy)

    print(f"{i+1:<6} {seed:<8} {f1_macro:<12.4f} {f1_per_class[0]:<12.4f} {f1_per_class[1]:<12.4f} {accuracy:<10.4f}")

# ========================================
# 3. AGGREGATE STATISTICS
# ========================================
print("\n" + "="*80)
print(f"DATASET 2 — AGGREGATE STATISTICS ({N_RUNS} RUNS)")
print("="*80)

def report(name, scores):
    mean = np.mean(scores)
    std = np.std(scores)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)
    print(f"{name:<18}: {mean:.4f} ± {std:.4f}   95% CI: [{ci_low:.4f}, {ci_high:.4f}]   Min/Max: [{min(scores):.4f}, {max(scores):.4f}]")

report("F1 (macro)", f1_macro_scores)
report("F1 (Healthy)", f1_class0_scores)
report("F1 (Parkinsonian)", f1_class1_scores)
report("Precision", precision_scores)
report("Recall", recall_scores)
report("Accuracy", accuracy_scores)

# ========================================
# 4. SAVE RESULTS
# ========================================
results_df = pd.DataFrame({
    'Run': range(1, N_RUNS+1),
    'Seed': range(N_RUNS),
    'F1_Macro': f1_macro_scores,
    'F1_Healthy': f1_class0_scores,
    'F1_Parkinsonian': f1_class1_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'Accuracy': accuracy_scores
})
results_df.to_csv("tstr_multiple_runs_dataset2_spirals.csv", index=False)
print(f"\n✓ Results saved to: tstr_multiple_runs_dataset2_spirals.csv")

np.save("tstr_f1_macro_dataset2.npy", np.array(f1_macro_scores))
print(f"✓ F1 array saved to: tstr_f1_macro_dataset2.npy")


# %% [Code cell 13]
#K-Fold

# Dataset 2 (Spiral)
df = pd.read_csv("extracted_featuresnewtest.csv")

df["target"] = df["Label"].map({"Healthy": 0, "Parkinsonian": 1})
X = df.drop(columns=["Label", "Image", "target"], errors="ignore")
y = df["target"]

# Repeated Stratified K-Fold (ΙΔΙΟ random_state=42 για συνέπεια)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

# Στατιστικά
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
ci_low, ci_high = np.percentile(f1_scores, [2.5, 97.5])

print(f"Dataset 2 :")
print(f"F1 = {mean_f1:.3f} ± {std_f1:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Min/Max: [{f1_scores.min():.3f}, {f1_scores.max():.3f}]")

#
np.save("rf_f1_dataset2.npy", f1_scores)


# %% [Code cell 14]
# Create folders if they don't exist
os.makedirs("figures", exist_ok=True)
os.makedirs("fuzzysets", exist_ok=True)

# Load data
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")
feature_columns = [col for col in df_synth.columns if col not in ['Image', 'Label']]

sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

for feature in feature_columns:
    plt.figure(figsize=(10, 5))

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Label', y=feature, data=df_synth)
    plt.title(f"Boxplot: {feature}")

    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x='Label', y=feature, data=df_synth, inner="box")
    plt.title(f"Violin Plot: {feature}")

    plt.tight_layout()
    plt.savefig(f"figures/{feature}_distribution.png")
    plt.show()


# %% [Code cell 15]
# === 1. Load Data ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")
os.makedirs("plots/miniplots", exist_ok=True)
os.makedirs("plots/projections", exist_ok=True)

# === 2. Get feature columns ===
feature_cols = [col for col in df.columns if col not in ['Image', 'Label']]

# === 3. Generate 10 diverse plots ===
for feature in feature_cols[:10]:  # adjust slice for more features
    plt.figure(figsize=(14, 8))

    # Histogram
    plt.subplot(2, 3, 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} - Histogram")

    # Boxplot
    plt.subplot(2, 3, 2)
    sns.boxplot(x='Label', y=feature, data=df)
    plt.title(f"{feature} - Boxplot")

    # Violin plot
    plt.subplot(2, 3, 3)
    sns.violinplot(x='Label', y=feature, data=df)
    plt.title(f"{feature} - Violin Plot")

    # Strip plot
    plt.subplot(2, 3, 4)
    sns.stripplot(x='Label', y=feature, data=df, jitter=True, alpha=0.5)
    plt.title(f"{feature} - Strip Plot")

    # KDE
    plt.subplot(2, 3, 5)
    sns.kdeplot(data=df, x=feature, hue='Label', fill=True)
    plt.title(f"{feature} - KDE by Label")

    # ECDF
    plt.subplot(2, 3, 6)
    for label in df['Label'].unique():
        subset = df[df['Label'] == label]
        sns.ecdfplot(subset[feature], label=label)
    plt.title(f"{feature} - ECDF")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/miniplots/{feature}_summary.png")
    plt.close()

# === 4. PCA 2D and 3D ===
X = df[feature_cols]
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
pca_df_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
pca_df_2d['Label'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df_2d, x='PC1', y='PC2', hue='Label')
plt.title("PCA - 2D Projection")
plt.savefig("plots/projections/pca_2d.png")
plt.close()

# PCA 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for label in pca_df_2d['Label'].unique():
    idx = pca_df_2d['Label'] == label
    ax.scatter(X_pca_3d[idx, 0], X_pca_3d[idx, 1], X_pca_3d[idx, 2], label=label)
ax.set_title("PCA - 3D Projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.savefig("plots/projections/pca_3d.png")
plt.close()

# === 5. t-SNE Visualization ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['Label'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='Label')
plt.title("t-SNE - 2D Projection")
plt.savefig("plots/projections/tsne_2d.png")
plt.close()

# === 6. Correlation Heatmap ===
plt.figure(figsize=(12, 10))
corr = df[feature_cols].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Features")
plt.savefig("plots/projections/correlation_heatmap.png")
plt.close()

# === 7. Pairplot (use subset for clarity) ===
sns.pairplot(df[feature_cols[:5] + ['Label']], hue='Label')
plt.savefig("plots/projections/pairplot.png")
plt.close()

print("✅ All visualizations have been saved to the 'plots/' directory.")


# %% [Code cell 17]
# === 1. Load real dataset ===
real_df = pd.read_csv("extracted_featuresnew.csv")
real_features = real_df.drop(columns=["Image", "Label"], errors='ignore')

# === 2. Scale real data with the same method ===
scaler = StandardScaler()
real_scaled = scaler.fit_transform(real_features)

# === 3. Train model again on synthetic data for prediction ===
# (Normally you'd save/load the trained model, but here we retrain for simplicity)

# Load synthetic data
synth_df = pd.read_csv("ctgan_true_positives_all1000.csv")
X_synth = synth_df.drop(columns=["Image", "Label"], errors='ignore')
y_synth = synth_df["Label"]
if y_synth.dtype == object:
    y_synth = y_synth.map({"Healthy": 0, "Parkinsonian": 1})

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_synth, y_synth)

# === 4. Predict on real data ===
y_pred_real = rf.predict(real_scaled)
proba_real = rf.predict_proba(real_scaled)

# === 5. Attach predictions back to dataframe ===
real_df["Predicted_Label"] = y_pred_real
real_df["Predicted_Label"] = real_df["Predicted_Label"].map({0: "Healthy", 1: "Parkinsonian"})

# === 6. Save output ===
os.makedirs("predictions", exist_ok=True)
real_df.to_csv("predictions/real_data_with_rf_predictions.csv", index=False)
print("✅ Saved predictions to predictions/real_data_with_rf_predictions.csv")

# === 7. Heatmap of prediction probabilities ===
plt.figure(figsize=(10, 6))
sns.heatmap(proba_real, cmap="viridis", xticklabels=["Healthy", "Parkinsonian"], cbar_kws={'label': 'Probability'})
plt.title("Prediction Probability Heatmap (Real Data)")
plt.xlabel("Predicted Class")
plt.ylabel("Sample Index")
plt.tight_layout()
plt.savefig("predictions/real_prediction_heatmap.png")
plt.show()


# %% [Code cell 18]
# === 1. Load real dataset ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")  # Adjust path if needed

# === 2. Selected features for fuzzy sets ===
features_to_fuzzify = [
    "Wave Energy", "Irregularities", "Signal Entropy",
    "Skewness", "Kurtosis", "Smoothness", "Irregularities", "Average Amplitude"
]

# === 3. Function to compute triangular fuzzy sets ===
def tight_fuzzy_sets(data, feature):
    values = data[feature].dropna()
    min_val = values.min()
    max_val = values.max()

    # Key percentiles (tight overlap)
    p20 = np.percentile(values, 20)
    p35 = np.percentile(values, 35)
    p40 = np.percentile(values, 40)
    p50 = np.percentile(values, 50)
    p60 = np.percentile(values, 60)
    p65 = np.percentile(values, 65)
    p80 = np.percentile(values, 80)

    return [
        {"Feature": feature, "Set": "Low",    "A": min_val, "B": p20, "C": p40},
        {"Feature": feature, "Set": "Medium", "A": p35,     "B": p50, "C": p65},
        {"Feature": feature, "Set": "High",   "A": p60,     "B": p80, "C": max_val}
    ]

# === 4. Generate fuzzy sets for all selected features ===
fuzzy_definitions = []
for feat in features_to_fuzzify:
    fuzzy_definitions.extend(tight_fuzzy_sets(df, feat))

# === 5. Save & display results ===
fuzzy_df = pd.DataFrame(fuzzy_definitions)
print("📊 Tight Fuzzy Sets with Minimal Overlap:")
print(fuzzy_df)

# Optional: save to CSV
fuzzy_df.to_csv("tight_fuzzy_sets.csv", index=False)
print("\n✅ Saved to: tight_fuzzy_sets.csv")


# %% [Code cell 19]
# === 1. Load your dataset ===
df = pd.read_csv("extracted_featuresnew.csv")  # Update this path if needed

# === 2. Define tight fuzzy sets (minimal overlap) ===
fuzzy_sets = {
    "Wave Energy": [
        ("Low", 917.765136, 1034.343286, 1211.038055),
        ("Medium", 1169.017630, 1287.489969, 1416.875960),
        ("High", 1359.815092, 1592.945928, 3519.630422)
    ],
    "Irregularities": [
        ("Low", 246.000000, 251.000000, 252.000000),
        ("Medium", 252.000000, 253.000000, 254.000000),
        ("High", 253.000000, 254.000000, 255.000000)
    ],
    "Signal Entropy": [
        ("Low", 2.249119, 2.272657, 2.284137),
        ("Medium", 2.282455, 2.286809, 2.289927),
        ("High", 2.288833, 2.293290, 2.299413)
    ],
    "Skewness": [
        ("Low", -0.357621, -0.080322, -0.017587),
        ("Medium", -0.031371, 0.014581, 0.065292),
        ("High", 0.045163, 0.120252, 0.430393)
    ],
    "Kurtosis": [
        ("Low", -1.300223, -1.191184, -1.131120),
        ("Medium", -1.146514, -1.106764, -1.054824),
        ("High", -1.075387, -0.988812, -0.835933)
    ],
    "Smoothness": [
        ("Low", 0.019822, 0.049614, 0.057431),
        ("Medium", 0.056119, 0.060259, 0.064683),
        ("High", 0.063055, 0.069708, 0.076755)
    ],
    "Average Amplitude": [
        ("Low", 0.020232, 0.053158, 0.064042),
        ("Medium", 0.062134, 0.067730, 0.072455),
        ("High", 0.071007, 0.077622, 0.083773)
    ]
}

# === 3. Triangular membership function ===
def get_membership_level(value, sets):
    best_label = None
    best_membership = -1
    for label, a, b, c in sets:
        if value <= a or value >= c:
            membership = 0
        elif a < value <= b:
            membership = (value - a) / (b - a)
        elif b < value < c:
            membership = (c - value) / (c - b)
        else:
            membership = 0
        if membership > best_membership:
            best_membership = membership
            best_label = label
    return best_label

# === 4. Apply fuzzy logic per feature ===
output_df = df[["Image"]].copy()
for feature in fuzzy_sets:
    output_df[feature + " Level"] = df[feature].apply(lambda val: get_membership_level(val, fuzzy_sets[feature]))

# === 5. Export result ===
output_df.to_csv("individuals_with_updated_fuzzy_levels.csv", index=False)
print("✅ Saved: individuals_with_updated_fuzzy_levels.csv")


# %% [Code cell 20]
# === Triangular membership function ===
def triangular_membership(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# === Plotting for selected features ===
selected_features = ["Smoothness", "Wave Energy", "Signal Entropy"]

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharey=True)

for ax, feature in zip(axes, selected_features):
    sets = fuzzy_sets[feature]

    # Plot each fuzzy set (Low, Medium, High)
    for label, a, b, c in sets:
        x = np.linspace(a, c, 300)
        y = [triangular_membership(val, a, b, c) for val in x]
        ax.plot(x, y, label=f"{label} ({a:.3f}, {b:.3f}, {c:.3f})")

    # Overlay actual values
    values = df[feature].dropna().values
    ax.scatter(values, np.zeros_like(values), alpha=0.3, color="black", s=15, label="Individuals")

    ax.set_title(f"Fuzzy Sets for {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Membership Degree")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("dataset2_fuzzy_sets.png", dpi=300)
plt.show()

print("✅ Exported fuzzy plots as: dataset2_fuzzy_sets.png")


# %% [Code cell 21]
# === 6. Plot fuzzy membership level distributions ===
fuzzy_plot_folder = "fuzzy_membership_plots"
os.makedirs(fuzzy_plot_folder, exist_ok=True)


# For each fuzzy-analyzed feature, plot membership distribution
for feature in fuzzy_sets:
    level_col = feature + " Level"
    plt.figure(figsize=(6, 4))
    sns.countplot(x=level_col, data=output_df, order=["Low", "Medium", "High"])
    plt.title(f"Fuzzy Membership Levels for {feature}")
    plt.xlabel("Membership Level")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(fuzzy_plot_folder, f"{feature}_fuzzy_levels.png")
    plt.savefig(plot_path)
    plt.close()


# %% [Code cell 22]
# === 6. Plot triangular membership functions ===
triangular_plot_folder = "triangular_membership_plots"
os.makedirs(triangular_plot_folder, exist_ok=True)


for feature, sets in fuzzy_sets.items():
    plt.figure(figsize=(6, 4))

    # Plot each fuzzy set for the current feature
    for label, a, b, c in sets:
        x = np.linspace(a, c, 500)
        y = np.piecewise(
            x,
            [x <= a, (x > a) & (x <= b), (x > b) & (x < c), x >= c],
            [0,
             lambda x: (x - a) / (b - a) if b != a else 1,
             lambda x: (c - x) / (c - b) if c != b else 1,
             0]
        )
        plt.plot(x, y, label=label)

    plt.title(f"Triangular Membership Functions for {feature}")
    plt.xlabel(feature)
    plt.ylabel("Membership Degree")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(triangular_plot_folder, f"{feature}_triangular.png")
    plt.savefig(plot_path, dpi=300,            # high-resolution
    bbox_inches="tight") # trims extra whitespace)
    plt.close()


# %% [Code cell 23]
# === 1. Load the full fuzzy-level dataset ===
df = pd.read_csv("individuals_with_updated_fuzzy_levels.csv")  # Update path if needed

# === 2. Infer true label from filename ===
def infer_true_label(filename):
    if 'H' in filename:
        return "Healthy"
    elif 'P' in filename:
        return "Parkinsonian"
    else:
        return "Unknown"

df["True Label"] = df["Image"].apply(infer_true_label)

# === 3. Filter only known ground truth ===
df = df[df["True Label"].isin(["Healthy", "Parkinsonian"])]

# === 4. Count unique fuzzy-level patterns per group ===
features = ["Smoothness Level", "Wave Energy Level", "Signal Entropy Level"]

# Get most frequent patterns for Healthy
healthy_df = df[df["True Label"] == "Healthy"]
healthy_patterns = healthy_df[features].value_counts().reset_index(name="Count")
healthy_patterns["Group"] = "Healthy"

# Get most frequent patterns for Parkinsonian
parkinsonian_df = df[df["True Label"] == "Parkinsonian"]
parkinsonian_patterns = parkinsonian_df[features].value_counts().reset_index(name="Count")
parkinsonian_patterns["Group"] = "Parkinsonian"

# Combine and show
rule_candidates = pd.concat([healthy_patterns, parkinsonian_patterns], ignore_index=True)
rule_candidates = rule_candidates.sort_values(by=["Group", "Count"], ascending=[True, False])

# === 5. Save the candidate rule patterns ===
rule_candidates.to_csv("candidate_fuzzy_rules_full_dataset.csv", index=False)
print("✅ Saved: candidate_fuzzy_rules_full_dataset.csv")


# %% [Code cell 24]
# ========================================
# DATASET 2 (SPIRALS) — THREE-STAGE STATISTICAL VALIDATION
# ========================================


# ========================================
# 1. LOAD DATA — DATASET 2
# ========================================
# Real test data for spirals
df_real = pd.read_csv("extracted_featuresnewtest.csv")

# Synthetic data from CTGAN (spirals)
# ⚠ ΕΛΕΓΞΕ: Είναι το ίδιο αρχείο με του Dataset 1; Αν ναι, ίσως είναι μεθοδολογικό issue
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")  # ή το spiral-specific synthetic CSV

# ========================================
# 2. SEPARATE BY CLASS
# ========================================
real_healthy = df_real[df_real['Label'] == 'Healthy'] if df_real['Label'].dtype == 'O' else df_real[df_real['Label'] == 0]
real_parkinson = df_real[df_real['Label'] == 'Parkinsonian'] if df_real['Label'].dtype == 'O' else df_real[df_real['Label'] == 1]

synth_healthy = df_synth[df_synth['Label'] == 'Healthy'] if df_synth['Label'].dtype == 'O' else df_synth[df_synth['Label'] == 0]
synth_parkinson = df_synth[df_synth['Label'] == 'Parkinsonian'] if df_synth['Label'].dtype == 'O' else df_synth[df_synth['Label'] == 1]

print(f"Real Healthy:           {len(real_healthy)} samples")
print(f"Real Parkinsonian:      {len(real_parkinson)} samples")
print(f"Synthetic Healthy:      {len(synth_healthy)} samples")
print(f"Synthetic Parkinsonian: {len(synth_parkinson)} samples")

# ========================================
# 3. FEATURE COLUMNS — Dataset 2 (Spirals)
# ========================================
exclude_cols = ["Image", "Label", "PatientID", "ID", "Unnamed: 0", "target", "Group"]
feature_cols = [col for col in df_real.columns if col not in exclude_cols]
print(f"\nFeatures to test: {feature_cols}\n")

# ========================================
# 4. THREE-STAGE VALIDATION FUNCTION
# ========================================
def three_stage_validation(real_data, synth_data, class_name, feature_cols, dataset_name="Dataset 2 (Spirals)"):
    print("="*100)
    print(f"{dataset_name} — VALIDATION PROTOCOL — {class_name.upper()} CLASS")
    print("="*100)

    results = []

    for col in feature_cols:
        if col not in real_data.columns or col not in synth_data.columns:
            print(f"\n📌 {col}: Skipped (not in both datasets)")
            continue

        real = real_data[col].dropna()
        synth = synth_data[col].dropna()

        if real.nunique() < 2 or synth.nunique() < 2:
            print(f"\n📌 {col}: Skipped (insufficient variation)")
            continue

        # STAGE 1: Shapiro-Wilk
        real_sample = real.sample(min(len(real), 5000), random_state=42) if len(real) > 5000 else real
        synth_sample = synth.sample(min(len(synth), 5000), random_state=42) if len(synth) > 5000 else synth

        sw_real_stat, sw_real_p = shapiro(real_sample)
        sw_synth_stat, sw_synth_p = shapiro(synth_sample)

        real_normal = sw_real_p > 0.05
        synth_normal = sw_synth_p > 0.05
        both_normal = real_normal and synth_normal

        # STAGE 2: Test Selection
        if both_normal:
            test_stat, test_p = ttest_ind(real, synth, equal_var=False)
            test_name = "Welch's t-test"
            test_label = "t"
        else:
            test_stat, test_p = mannwhitneyu(real, synth, alternative='two-sided')
            test_name = "Mann-Whitney U"
            test_label = "U"

        # STAGE 2b: Wasserstein
        wd = wasserstein_distance(real, synth)

        # STAGE 3: Interpretation
        same_dist = test_p > 0.05

        print(f"\n📌 {col}")
        print(f"   Shapiro-Wilk Real:       p = {sw_real_p:.4f} → {'Normal ✓' if real_normal else 'NOT Normal ⚠'}")
        print(f"   Shapiro-Wilk Synthetic:  p = {sw_synth_p:.4f} → {'Normal ✓' if synth_normal else 'NOT Normal ⚠'}")
        print(f"   ➔ {'Both normal' if both_normal else 'At least one non-normal'} → using {test_name}")
        print(f"   {test_name}:           {test_label}-stat = {test_stat:.4f}, p = {test_p:.4f}")
        print(f"   Wasserstein Distance:    {wd:.4f}")
        print(f"   FINAL: {'Same Distribution ✓' if same_dist else 'Different ⚠'}")

        results.append({
            'Dataset': dataset_name,
            'Class': class_name,
            'Feature': col,
            'N_Real': len(real),
            'N_Synthetic': len(synth),
            'Shapiro_Real_p': round(sw_real_p, 4),
            'Shapiro_Synth_p': round(sw_synth_p, 4),
            'Real_Normal': real_normal,
            'Synth_Normal': synth_normal,
            'Both_Normal': both_normal,
            'Test_Used': test_name,
            'Test_Statistic': round(test_stat, 4),
            'Test_p_value': round(test_p, 4),
            'Wasserstein_Distance': round(wd, 4),
            'Same_Distribution': same_dist
        })

    return results

# ========================================
# 5. RUN FOR BOTH CLASSES
# ========================================
healthy_results = three_stage_validation(real_healthy, synth_healthy, "Healthy", feature_cols)
print("\n")
parkinson_results = three_stage_validation(real_parkinson, synth_parkinson, "Parkinsonian", feature_cols)

# ========================================
# 6. CONSOLIDATED RESULTS
# ========================================
all_results = healthy_results + parkinson_results
results_df = pd.DataFrame(all_results)

results_df.to_csv("statistical_validation_dataset2_spirals.csv", index=False)

print("\n" + "="*100)
print("DATASET 2 (SPIRALS) — CONSOLIDATED RESULTS")
print("="*100)
print(results_df[['Class', 'Feature', 'Test_Used', 'Test_p_value', 'Wasserstein_Distance', 'Same_Distribution']].to_string(index=False))

n_total = len(all_results)
n_same = sum(r['Same_Distribution'] for r in all_results)

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"Total comparisons: {n_total}")
print(f"Same distribution: {n_same} ({100*n_same/n_total:.1f}%)")
print(f"\n✓ Results saved to: statistical_validation_dataset2_spirals.csv")


