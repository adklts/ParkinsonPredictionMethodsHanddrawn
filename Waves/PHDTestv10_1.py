"""
Exported from PHDTestv10_1.ipynb.
Repeated imports were consolidated at the top.
Jupyter shell/magic commands were converted into comments.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import shap
import warnings
import joblib
import matplotlib.transforms as transforms
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
    wasserstein_distance
)
from scipy.fftpack import fft
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from matplotlib.table import Table
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse, Patch
from IPython.display import display


# %% Cell 0
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
#healthy_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\wave\training\Healthy'
#parkinsonian_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\wave\training\Parkinson'
healthy_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\wave\testing\Healthy'
parkinsonian_folder = r'C:\Users\adklt\OneDrive - aegean.gr\Desktop\wave\testing\Parkinson'

# === Process Images ===
healthy_data = process_folder(healthy_folder, "Healthy")
parkinsonian_data = process_folder(parkinsonian_folder, "Parkinsonian")
all_data = healthy_data + parkinsonian_data
df = pd.DataFrame(all_data)

# === Save and Preview ===
#df.to_csv("extracted_featuresnewonly.csv", index=False)
df.to_csv("extracted_featuresnewonlytesting.csv", index=False)
print("Saved to extracted_features.csv")
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
    plt.show()
    
# Create a directory to save the plots
output_dir = "boxplots"
os.makedirs(output_dir, exist_ok=True)

# Generate and save boxplots
features_to_plot = [col for col in df.columns if col not in ['Image', 'Label']]
for feature in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Label', y=feature, data=df)
    plt.title(f"Boxplot of {feature}")
    plt.tight_layout()

    # Save the plot as a JPEG file
    filename = os.path.join(output_dir, f"boxplot_{feature}.jpeg")
    plt.savefig(filename, format='jpeg', dpi=300)
    plt.close()  # Close the figure to free up memory

# %% Cell 1
# === Descriptive Statistics for Each Feature by Label ===
print("\n=== Full Descriptive Statistics by Label ===")

# Get grouped descriptive stats
grouped_stats = df.groupby("Label")[features_to_plot].describe()

# Loop through features and print cleanly
for feature in features_to_plot:
    print(f"\n📊 === {feature} ===")
    display(grouped_stats[feature])  # works in Jupyter / shows nicely in console

# %% Cell 3
# Jupyter shell/magic command removed from executable script: !pip install sdv

# %% Cell 4
#!pip install pandas

# Jupyter shell/magic command removed from executable script: !{sys.executable} -m pip install pandas

# %% Cell 5
# Jupyter shell/magic command removed from executable script: !{sys.executable} -m pip install sdv

# %% Cell 9
#Enough good 18/4/2025 my last use

# -----------------------------------------------------
# ⚙️ CONFIGURATION
# -----------------------------------------------------
USE_TVAE = False  # Set to True to try TVAE instead of CTGAN
SYNTH_PER_CLASS = 1000
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

output_file = "ctgan_synthetic_1000_per_class.csv"
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
        df_true_positives.to_csv("ctgan_true_positives_all1000.csv", index=False)

        # Save per-label TPs
        for label in ["Healthy", "Parkinsonian"]:
            df_tp_label = df_true_positives[df_true_positives["Label"] == label]
            out = f"ctgan_truepositives_{label.lower()}.csv"
            df_tp_label.to_csv(out, index=False)
            print(f"💾 Saved {len(df_tp_label)} True Positives for {label} → {out}")

# %% Cell 11
# === Load real and synthetic ===
df_real = pd.read_csv("extracted_featuresnew.csv")
# df_synth = pd.read_csv("ctgan_synthetic_1000_per_class.csv")
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")

# === Match columns ===
feature_cols = [col for col in df_real.columns if col not in ["Image", "Label"]]

# === Ensure Label is same format ===
df_real["Origin"] = "Real"
df_synth["Origin"] = "Synthetic"
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# === Create output folder ===
out_dir = "comparision"
os.makedirs(out_dir, exist_ok=True)

# === 1. Boxplots per feature ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Origin", y=col, data=df_combined)
    plt.title(f"📦 Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"boxplot_{col}.png"), dpi=300)
    plt.close()

# === 2. Overlaid histograms ===
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_real[col], color="blue", label="Real", kde=True, stat="density")
    sns.histplot(df_synth[col], color="orange", label="Synthetic", kde=True, stat="density")
    plt.title(f"📈 Histogram: {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"histogram_{col}.png"), dpi=300)
    plt.close()

# === 3. T-test comparison per feature ===
for col in feature_cols:
    if df_real[col].nunique() < 2 or df_synth[col].nunique() < 2:
        print(f"{col:25}: Skipped (not enough variation)")
        continue

    try:
        t_stat, p_value = ttest_ind(df_real[col], df_synth[col], equal_var=False)
        result = "Same Distribution ✅" if p_value > 0.05 else "Different ⚠️"

        italic_p = "\U0001D45D"  # Unicode italic p
        print(f"{col:25}: {italic_p} = {p_value:.4f} → {result}")

    except Exception as e:
        print(f"{col:25}: Error — {e}")

# %% Cell 12
# === Load real and synthetic ===
df_real = pd.read_csv("extracted_featuresnew.csv")
## df_synth = pd.read_csv("ctgan_synthetic_1000_per_class.csv")

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

# %% Cell 13
# === Load data ===
df_real = pd.read_csv("extracted_featuresnew.csv")
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")

# === Match columns ===
feature_cols = [col for col in df_real.columns if col not in ["Image", "Label"]]

# === Add origin labels ===
df_real["Origin"] = "Real"
df_synth["Origin"] = "Synthetic"
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# === Plot with improved style ===
for col in feature_cols:
    if df_real[col].nunique() < 2 or df_synth[col].nunique() < 2:
        print(f"{col:25}: Skipped (not enough variation)")
        continue

    # Welch’s t-test
    t_stat, p_value = ttest_ind(df_real[col], df_synth[col], equal_var=False)

    # Significance stars
    if p_value < 0.001:
        sig_label = "***"
    elif p_value < 0.01:
        sig_label = "**"
    elif p_value < 0.05:
        sig_label = "*"
    else:
        sig_label = "ns"

    # Create figure
    plt.figure(figsize=(7, 5))
    ax = sns.violinplot(x="Origin", y=col, data=df_combined, inner=None, palette=["#1f77b4", "#ff7f0e"])
    sns.boxplot(x="Origin", y=col, data=df_combined, width=0.2, showcaps=True, boxprops={'facecolor':'white'}, 
                showfliers=False, whiskerprops={'linewidth':2}, ax=ax)

    # Title and labels
    plt.title(f"{col}: Real vs Synthetic", fontsize=16)
    plt.xlabel("Dataset Origin", fontsize=14)
    plt.ylabel(col, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add sample size text
    n_real, n_synth = len(df_real), len(df_synth)
    plt.text(0, df_real[col].mean(), f"N={n_real}", ha='center', va='bottom', fontsize=11, color="blue")
    plt.text(1, df_synth[col].mean(), f"N={n_synth}", ha='center', va='bottom', fontsize=11, color="orange")

    # Add p-value / significance annotation
    ymax = max(df_combined[col])
    y_offset = (ymax * 0.05)
    plt.text(0.5, ymax + y_offset, f"p={p_value:.3e} {sig_label}", ha='center', fontsize=12, color="red")

    plt.tight_layout()
    plt.show()

# %% Cell 14
# --- data ---
df_real  = pd.read_csv("extracted_featuresnew.csv")
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")
feature_cols = [c for c in df_real.columns if c not in ["Image","Label"]]

# --- collect stats ---
rows = []
for col in feature_cols:
    if df_real[col].nunique()<2 or df_synth[col].nunique()<2:
        rows.append({"Feature": col, "p": np.nan, "Status": "Skipped (low variance)"})
        continue
    t,p = ttest_ind(df_real[col], df_synth[col], equal_var=False)
    rows.append({"Feature": col, "p": p, "Status": "Same dist" if p>0.05 else "Different"})

res = pd.DataFrame(rows)
res_sig = res.dropna().sort_values("p")               # tested features
res_skip = res[res["p"].isna()]                       # skipped features
res_sig["neglog10p"] = -np.log10(res_sig["p"])        # spread small p's

# --- figure: horizontal bar of -log10(p) with cutoff ---
plt.figure(figsize=(8, max(4, 0.45*len(res_sig))))
ax = sns.barplot(y="Feature", x="neglog10p", data=res_sig)
ax.axvline(-np.log10(0.05), ls="--", lw=1.5)          # significance threshold
ax.set_xlabel(r"$-\,\log_{10}(p)$", fontsize=13)
ax.set_ylabel("")
ax.set_title("Welch’s t-test: Real vs. Synthetic (Dataset-1)", fontsize=15)
for i, p in enumerate(res_sig["p"]):
    ax.text(res_sig["neglog10p"].iloc[i]+0.03, i, f"p={p:.3e}",
            va="center", fontsize=10)

plt.tight_layout()

# --- add an inset table for skipped features (if any) ---
if len(res_skip):
    inset = plt.axes([0.62, 0.05, 0.35, 0.35])  # [left, bottom, width, height]
    inset.axis("off")
    tbl = inset.table(cellText=[[f] for f in res_skip["Feature"]],
                      colLabels=["Skipped (low variance)"],
                      loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.2)

plt.show()

# %% Cell 16
# tidy results table
tbl = res.copy()
tbl["p_display"] = tbl["p"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
tbl["Significance"] = np.where(tbl["p"].isna(), "—",
                               np.where(tbl["p"]<0.001, "***",
                               np.where(tbl["p"]<0.01,  "**",
                               np.where(tbl["p"]<0.05,  "*", "ns"))))
tbl = tbl[["Feature","p_display","Significance","Status"]]

# save for the manuscript
tbl.to_csv("Table_S1_ttest_results.csv", index=False)

# %% Cell 17
# Load the synthetic dataset (if not already loaded)
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")

# Get list of feature columns (exclude Image and Label)
feature_columns = [col for col in df_synth.columns if col not in ['Image', 'Label']]

# Group by Label and describe each feature
grouped_stats = df_synth.groupby("Label")[feature_columns].describe()

# Display per feature
for feature in feature_columns:
    print(f"\n📊 === {feature} ===")
    print(grouped_stats[feature])

# %% Cell 18
# Jupyter shell/magic command removed from executable script: !pip install shap

# %% Cell 19
warnings.filterwarnings("ignore")  # Optional: silence warnings from SHAP/Matplotlib

# === 1. Load data ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")

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

# === 8. SHAP Explanation ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (for tree-based binary classifier), grab class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# If still 3D, reduce to 2D using class index or mean
if shap_values.ndim == 3:
    # Example: use class 1 (assuming it's Parkinsonian)
    shap_values_reduced = shap_values[:, :, 1]
else:
    shap_values_reduced = shap_values

# Now shap_values_reduced should be (n_samples, n_features)
print(f"Final SHAP shape: {shap_values_reduced.shape}, X_test shape: {X_test.shape}")

# Plot
shap.summary_plot(shap_values_reduced, X_test, plot_type="bar")
shap.summary_plot(shap_values_reduced, X_test)




# Summary bar plot (global feature importance)
#shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Optional: detailed beeswarm plot
#shap.summary_plot(shap_values[1], X_test)

green_features = ["Smoothness", "Kurtosis", "Signal Entropy", "Wave Energy", "Irregularities", "Skewness"]

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
    
plt.savefig("shap_beeswarm.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.clf()

# %% Cell 20
# --- Figure 2A: SHAP beeswarm (dot) ---
shap.summary_plot(
    shap_values_reduced, X_test,
    plot_type="dot", show=False, max_display=10, color_bar=True
)
fig = plt.gcf()
fig.set_size_inches(8, 6)
# label the colorbar
cb_ax = fig.axes[-1]
cb_ax.set_ylabel("Feature value (low → high)", fontsize=12)
plt.title("SHAP summary: impact on model output", fontsize=16)
plt.xlabel("SHAP value (impact on model output)", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
fig.savefig("figure2_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Figure 2B: SHAP bar (global importance) ---
shap.summary_plot(
    shap_values_reduced, X_test,
    plot_type="bar", show=False, max_display=10
)
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.title("Mean |SHAP| feature importance", fontsize=16)
plt.xlabel("Mean |SHAP value|", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
fig.savefig("figure2_shap_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% Cell 21
# --- Bar Summary Plot ---
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Save the plot to a JPEG file
plt.savefig("shap_summary_bar.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.clf()  # Clear the plot for the next one

# --- Optional: Beeswarm Plot ---
shap.summary_plot(shap_values[1], X_test)

# %% Cell 22
# === 8. Save the trained model ===
joblib.dump(model, "random_forest_ctgan_model.pkl")
print("💾 Model saved as random_forest_ctgan_model.pkl")

# ---------------------------------------------------
# 🧪 Apply model to extracted real data
# ---------------------------------------------------

# === Load real data ===
#df_real = pd.read_csv("extracted_featuresnew.csv") 
df_real = pd.read_csv("extracted_featuresnewonlytesting.csv")
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

# %% Cell 23
# ========================================
# DATASET 1 (WAVES) — MULTIPLE TSTR RUNS
# ========================================


# ========================================
# 1. LOAD DATA
# ========================================
# Synthetic data (CTGAN-generated, used for training)
synthetic_df = pd.read_csv("ctgan_true_positives_all1000.csv")

# Real test data (held-out test set)
df_real = pd.read_csv("extracted_featuresnewonlytesting.csv")

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
print(f"DATASET 1 (WAVES) — TSTR EVALUATION — {N_RUNS} RUNS")
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
print(f"DATASET 1 — AGGREGATE STATISTICS ({N_RUNS} RUNS)")
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
results_df.to_csv("tstr_multiple_runs_dataset1_waves.csv", index=False)
print(f"\n✓ Results saved to: tstr_multiple_runs_dataset1_waves.csv")

np.save("tstr_f1_macro_dataset1.npy", np.array(f1_macro_scores))
print(f"✓ F1 array saved to: tstr_f1_macro_dataset1.npy")

# %% Cell 24
"""
═══════════════════════════════════════════════════════════════════
ΣΤΑΤΙΣΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ML — Dataset 1 (Waveforms)
Repeated Stratified K-Fold Cross-Validation
═══════════════════════════════════════════════════════════════════

Στόχος: αντικατάσταση του single F1 = 0.80 με
        mean F1 ± 95% CI υπολογισμένο σε 50 iterations
        
Πρωτόκολλο:
  - 5-fold × 10 repeats = 50 evaluations
  - Stratified (διατηρεί την αναλογία Healthy/Parkinsonian)
  - Εκπαίδευση μόνο σε real data (καθαρό baseline)
  - Optionally: επανάληψη με CTGAN augmentation ως ablation
"""


# ═══════════════════════════════════════════════════════════════
# CELL 1: Load data
# ═══════════════════════════════════════════════════════════════

# ΠΡΟΣΟΧΗ: Βάλε εδώ το ΠΛΗΡΕΣ real dataset (όχι μόνο το test)
# Για να γίνει CV, χρειαζόμαστε όλα τα δείγματα
df = pd.read_csv("extracted_features_FULL_real.csv")  # ΑΛΛΑΞΕ ΤΟ ΑΝ ΧΡΕΙΑΖΕΤΑΙ

# Αν δεν έχεις «ενοποιημένο» αρχείο, ένωσε train + test:
# df_train = pd.read_csv("extracted_featuresnew_training.csv")
# df_test  = pd.read_csv("extracted_featuresnewonlytesting.csv")
# df = pd.concat([df_train, df_test], ignore_index=True)

print(f"Συνολικά δείγματα: {len(df)}")
print(f"Κατανομή κλάσεων:")
print(df["Label"].value_counts())

# Προετοιμασία X, y
df["target"] = df["Label"].map({"Healthy": 0, "Parkinsonian": 1})
X = df.drop(columns=["Label", "Image", "target"], errors="ignore")
y = df["target"]

print(f"\nFeatures: {list(X.columns)}")
print(f"Shape: X={X.shape}, y={y.shape}")


# ═══════════════════════════════════════════════════════════════
# CELL 2: Repeated Stratified K-Fold CV
# ═══════════════════════════════════════════════════════════════

# Setup CV — ΙΔΙΟ random_state για να μπορείς αργότερα
# να συγκρίνεις με LLM / Fuzzy Ontology σε ζευγαρωτό test
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

# Storage για όλες τις μετρικές
results = {
    "f1_macro": [], "f1_healthy": [], "f1_parkinsonian": [],
    "accuracy": [], "precision_macro": [], "recall_macro": [],
}

# Trace των confusion matrices για aggregate
cm_total = np.zeros((2, 2), dtype=int)

print(f"\n{'='*60}")
print(f"Τρέχω {N_SPLITS}-fold × {N_REPEATS} repeats = {N_SPLITS*N_REPEATS} iterations...")
print(f"{'='*60}\n")

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Εκπαίδευση (ΙΔΙΕΣ παράμετροι με το αρχικό μοντέλο)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Μετρικές
    results["f1_macro"].append(f1_score(y_test, y_pred, average='macro'))
    results["f1_healthy"].append(f1_score(y_test, y_pred, pos_label=0))
    results["f1_parkinsonian"].append(f1_score(y_test, y_pred, pos_label=1))
    results["accuracy"].append(accuracy_score(y_test, y_pred))
    results["precision_macro"].append(precision_score(y_test, y_pred, average='macro'))
    results["recall_macro"].append(recall_score(y_test, y_pred, average='macro'))
    
    cm_total += confusion_matrix(y_test, y_pred)
    
    if (fold_idx + 1) % 10 == 0:
        print(f"  Iteration {fold_idx+1}/50  — running mean F1: "
              f"{np.mean(results['f1_macro']):.3f}")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Στατιστικά + 95% Confidence Interval
# ═══════════════════════════════════════════════════════════════

def summarize(name, scores):
    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    return {
        "metric": name,
        "mean": mean, "std": std,
        "ci_low": ci_low, "ci_high": ci_high,
        "min": arr.min(), "max": arr.max(),
    }

print(f"\n\n{'='*70}")
print(f"ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ — Dataset 1 (Random Forest, 50 iterations)")
print(f"{'='*70}\n")

print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'Range':>20}")
print("-" * 76)
for metric, scores in results.items():
    s = summarize(metric, scores)
    print(f"{s['metric']:<20} "
          f"{s['mean']:>8.3f} "
          f"{s['std']:>8.3f} "
          f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]  "
          f"[{s['min']:.3f}, {s['max']:.3f}]")

# Confusion matrix αθροιστική
print(f"\nAggregate Confusion Matrix (50 folds combined):")
print(f"                    Pred Healthy   Pred Parkinsonian")
print(f"True Healthy        {cm_total[0,0]:>12}   {cm_total[0,1]:>17}")
print(f"True Parkinsonian   {cm_total[1,0]:>12}   {cm_total[1,1]:>17}")


# ═══════════════════════════════════════════════════════════════
# CELL 4: Σώσιμο των 50 F1 scores για μελλοντική σύγκριση
# ═══════════════════════════════════════════════════════════════

# ΣΗΜΑΝΤΙΚΟ: αυτό το αρχείο θα το χρειαστείς για το Wilcoxon test
# όταν θα συγκρίνεις με LLM και Fuzzy Ontology
np.save("ml_f1_scores_dataset1.npy", np.array(results["f1_macro"]))
print(f"\n✓ Αποθηκεύτηκαν τα 50 F1 scores στο 'ml_f1_scores_dataset1.npy'")

# Pandas DataFrame για eύκολη επεξεργασία
results_df = pd.DataFrame(results)
results_df.to_csv("cv_results_dataset1.csv", index=False)
print(f"✓ Αποθηκεύτηκε αναλυτικός πίνακας στο 'cv_results_dataset1.csv'")


# ═══════════════════════════════════════════════════════════════
# CELL 5: Οπτικοποίηση (boxplot + histogram)
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot όλων των metrics
metric_data = [results["accuracy"], results["f1_macro"], 
               results["precision_macro"], results["recall_macro"]]
metric_labels = ["Accuracy", "F1 Macro", "Precision Macro", "Recall Macro"]

bp = axes[0].boxplot(metric_data, labels=metric_labels, patch_artist=True,
                      boxprops=dict(facecolor='#0F766E', alpha=0.6),
                      medianprops=dict(color='#B91C1C', linewidth=2))
axes[0].set_ylabel("Score")
axes[0].set_title("Dataset 1 — Random Forest (50 CV iterations)", fontweight='bold')
axes[0].axhline(0.80, color='gray', linestyle='--', alpha=0.5,
                label="Original single-split F1 = 0.80")
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Histogram of F1 macro
f1_arr = np.array(results["f1_macro"])
axes[1].hist(f1_arr, bins=15, color='#0F766E', alpha=0.7, edgecolor='black')
axes[1].axvline(f1_arr.mean(), color='#B91C1C', linewidth=2, 
                label=f"Mean = {f1_arr.mean():.3f}")
axes[1].axvline(np.percentile(f1_arr, 2.5), color='gray', linestyle='--',
                label=f"95% CI lower = {np.percentile(f1_arr, 2.5):.3f}")
axes[1].axvline(np.percentile(f1_arr, 97.5), color='gray', linestyle='--',
                label=f"95% CI upper = {np.percentile(f1_arr, 97.5):.3f}")
axes[1].set_xlabel("F1 Macro Score")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of F1 across 50 CV folds", fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("cv_results_dataset1.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✓ Γράφημα αποθηκεύτηκε ως 'cv_results_dataset1.png'")


# ═══════════════════════════════════════════════════════════════
# CELL 6: Τι να γράψεις στο paper / διαφάνεια
# ═══════════════════════════════════════════════════════════════

f1_arr = np.array(results["f1_macro"])
acc_arr = np.array(results["accuracy"])

print(f"\n{'='*70}")
print("ΓΙΑ ΤΟ PAPER / ΔΙΑΦΑΝΕΙΑ — copy-paste ready:")
print(f"{'='*70}")
print(f"""
ML (Random Forest) στο Dataset 1:
  F1 macro: {f1_arr.mean():.3f} ± {f1_arr.std():.3f} 
            (95% CI: [{np.percentile(f1_arr,2.5):.3f}, {np.percentile(f1_arr,97.5):.3f}])
  Accuracy: {acc_arr.mean():.3f} ± {acc_arr.std():.3f} 
            (95% CI: [{np.percentile(acc_arr,2.5):.3f}, {np.percentile(acc_arr,97.5):.3f}])
  
  Πρωτόκολλο: 5-fold × 10 repeats Stratified Cross-Validation
              (50 iterations), Random Forest με 100 trees,
              random_state=42 για αναπαραγωγιμότητα.
""")

# %% Cell 25
#K-Fold


# Φόρτωσε ΟΛΑ τα πραγματικά δεδομένα (όχι μόνο test set)
df = pd.read_csv("extracted_featuresnew.csv")  # πλήρες real dataset
df["target"] = df["Label"].map({"Healthy": 0, "Parkinsonian": 1})
X = df.drop(columns=["Label", "Image", "target"], errors="ignore")
y = df["target"]

# Repeated Stratified K-Fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

# Στατιστικά
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
ci_low, ci_high = np.percentile(f1_scores, [2.5, 97.5])

print(f"F1 = {mean_f1:.3f} ± {std_f1:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Min/Max: [{f1_scores.min():.3f}, {f1_scores.max():.3f}]")

# Σώσε τις 50 τιμές για το βήμα 3 (statistical comparison)
np.save("rf_f1_dataset1.npy", f1_scores)

# %% Cell 26
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

# %% Cell 27
# === 1. Load Data ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")
os.makedirs("plots1/miniplots", exist_ok=True)
os.makedirs("plots1/projections", exist_ok=True)

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
    plt.savefig(f"plots1/miniplots/{feature}_summary.png")
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
plt.savefig("plots1/projections/pca_2d.png")
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
plt.savefig("plots1/projections/pca_3d.png")
plt.close()

# === 5. t-SNE Visualization ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['Label'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='Label')
plt.title("t-SNE - 2D Projection")
plt.savefig("plots1/projections/tsne_2d.png")
plt.close()

# === 6. Correlation Heatmap ===
plt.figure(figsize=(12, 10))
corr = df[feature_cols].corr()

# Create the heatmap with rotated tick labels
ax = sns.heatmap(corr, cmap='coolwarm', annot=False, cbar=True, square=True,
                 xticklabels=True, yticklabels=True)

# Rotate axis labels for readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap of Features")

# Adjust layout to prevent clipping
plt.tight_layout()

# Save to file
plt.savefig("plots1/projections/correlation_heatmap.png", dpi=300)
plt.close()

# === 7. Pairplot (use subset for clarity) ===
sns.pairplot(df[feature_cols[:5] + ['Label']], hue='Label')
plt.savefig("plots1/projections/pairplot.png")
plt.close()

print("✅ All visualizations have been saved to the 'plots/' directory.")

# %% Cell 28
# === 3. Generate 10 diverse plots (with fixes) ===
palette_lbl = {"Healthy": "blue", "Parkinsonian": "orange"}

for feature in feature_cols[:10]:  # adjust slice for more features
    fig = plt.figure(figsize=(14, 8))

    # consistent ranges for this feature
    x_min, x_max = df[feature].min(), df[feature].max()
    y_min, y_max = x_min, x_max  # for y-axis on label-based plots

    # (a) Histogram (two groups)
    ax1 = plt.subplot(2, 3, 1)
    sns.histplot(
        data=df, x=feature, hue="Label", kde=True, element="step",
        palette=palette_lbl, alpha=0.5
    )
    ax1.set_title(f"{feature} - Histogram")
    ax1.set_xlim(x_min, x_max)

    # (b) Boxplot
    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(x='Label', y=feature, data=df, palette=palette_lbl)
    ax2.set_title(f"{feature} - Boxplot")
    ax2.set_ylim(y_min, y_max)

    # (c) Violin plot
    ax3 = plt.subplot(2, 3, 3)
    sns.violinplot(x='Label', y=feature, data=df, palette=palette_lbl, inner="box")
    ax3.set_title(f"{feature} - Violin Plot")
    ax3.set_ylim(y_min, y_max)

    # (d) Strip plot
    ax4 = plt.subplot(2, 3, 4)
    sns.stripplot(x='Label', y=feature, data=df, palette=palette_lbl, jitter=True, alpha=0.6)
    ax4.set_title(f"{feature} - Strip Plot")
    ax4.set_ylim(y_min, y_max)

    # (e) KDE (two groups)
    ax5 = plt.subplot(2, 3, 5)
    sns.kdeplot(data=df, x=feature, hue='Label', fill=True, common_norm=False, alpha=0.4,
                palette=palette_lbl)
    ax5.set_title(f"{feature} - KDE by Label")
    ax5.set_xlim(x_min, x_max)

    # (f) ECDF (two groups)
    ax6 = plt.subplot(2, 3, 6)
    sns.ecdfplot(data=df, x=feature, hue='Label', palette=palette_lbl)
    ax6.set_title(f"{feature} - ECDF")
    ax6.set_xlim(x_min, x_max)

    # add panel letters (a)-(f)
    for i, ax in enumerate(fig.axes, start=1):
        ax.text(-0.12, 1.08, f"({chr(96+i)})", transform=ax.transAxes,
                size=12, weight='bold', va='bottom', ha='left')

    plt.tight_layout()
    plt.savefig(f"plots1/miniplots/{feature}_summary.png", dpi=300)
    plt.close()

# %% Cell 29
# === 4. PCA 2D and 3D (reads your CSV, saves high-DPI figure) ===

CSV_PATH = "ctgan_true_positives_all1000.csv"  # <-- your file
os.makedirs("plots/projections", exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)

# Features/labels
feature_cols = [c for c in df.columns if c not in ["Image", "Label"]]
X = df[feature_cols].values
y = df["Label"].values

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100  # %

# Helper: 95% confidence ellipse
def confidence_ellipse(x, y, ax, n_std=1.96, edgecolor="k", lw=1.3):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(1 - pearson)
    e = Ellipse((0, 0), width=ell_rx*2, height=ell_ry*2, fill=False,
                edgecolor=edgecolor, lw=lw)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    tr = transforms.Affine2D().rotate_deg(45).scale(sx, sy).translate(np.mean(x), np.mean(y))
    e.set_transform(tr + ax.transData)
    ax.add_patch(e)

PALETTE = {"Healthy": "#1f77b4", "Parkinsonian": "#ff7f0e"}
mask_h = (y == "Healthy")
mask_p = ~mask_h

plt.figure(figsize=(12, 5))

# (A) 2D projection with 95% ellipses
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(X_pca[mask_h, 0], X_pca[mask_h, 1], s=18, alpha=0.75,
            label="Healthy", color=PALETTE["Healthy"])
ax1.scatter(X_pca[mask_p, 0], X_pca[mask_p, 1], s=18, alpha=0.75,
            label="Parkinsonian", color=PALETTE["Parkinsonian"])
confidence_ellipse(X_pca[mask_h, 0], X_pca[mask_h, 1], ax1, edgecolor=PALETTE["Healthy"])
confidence_ellipse(X_pca[mask_p, 0], X_pca[mask_p, 1], ax1, edgecolor=PALETTE["Parkinsonian"])
ax1.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=12)
ax1.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=12)
ax1.set_title("PCA — 2D projection with 95% ellipses", fontsize=13)
ax1.legend(frameon=True)

# --- add subfigure label (A) inside the plot ---  # NEW
ax1.text(0.02, 0.98, "(A)", transform=ax1.transAxes,
         fontsize=14, fontweight="bold", va="top", ha="left")

# (B) 3D projection
ax2 = plt.subplot(1, 2, 2, projection="3d")
ax2.scatter(X_pca[mask_h, 0], X_pca[mask_h, 1], X_pca[mask_h, 2],
            s=12, alpha=0.75, label="Healthy", color=PALETTE["Healthy"])
ax2.scatter(X_pca[mask_p, 0], X_pca[mask_p, 1], X_pca[mask_p, 2],
            s=12, alpha=0.75, label="Parkinsonian", color=PALETTE["Parkinsonian"])
ax2.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax2.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax2.set_zlabel(f"PC3 ({var[2]:.1f}%)")
ax2.set_title("PCA — 3D projection", fontsize=13)
ax2.view_init(elev=18, azim=40)
ax2.legend(
    loc="upper right",            # anchor point
    bbox_to_anchor=(1.25, 1.05),  # x=1.25 shifts right, y=1.05 shifts up
    borderaxespad=0.,
    frameon=True
)

ax2.text2D(
    0.02, 0.98, "(B)", transform=ax2.transAxes,
    fontsize=14, fontweight="bold", va="top", ha="left",
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5),
    zorder=100
)

plt.tight_layout()
plt.savefig("plots/projections/figure6_pcaNew.png", dpi=300, bbox_inches="tight")
plt.close()

# %% Cell 30
# === Imports ===


sns.set(style="whitegrid", context="talk")

# === 1) Load data ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")

# Output folders
os.makedirs("plots1/miniplots", exist_ok=True)
os.makedirs("plots1/projections", exist_ok=True)

# === 2) Normalize labels so coloring is consistent everywhere ===
df["Label"] = (
    df["Label"].astype(str).str.strip().replace({
        "PD": "Parkinsonian",
        "Parkinson": "Parkinsonian",
        "parkinson": "Parkinsonian",
        "parkinsonian": "Parkinsonian",
        "healthy": "Healthy"
    })
)
df["Label"] = pd.Categorical(df["Label"], categories=["Healthy", "Parkinsonian"])
print("Unique labels after normalization:", df["Label"].unique())

# Feature columns (drop non-numeric id/label-like fields)
exclude_cols = {"Image", "Label"}
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

# Fixed palette
palette_lbl = {"Healthy": "blue", "Parkinsonian": "orange"}

def feature_xlim(series, trim_quantiles=(0.01, 0.99)):
    """
    Consistent limits per feature with optional outlier trimming.
    Set trim_quantiles=None to use full min/max.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if trim_quantiles:
        lo, hi = s.quantile(trim_quantiles[0]), s.quantile(trim_quantiles[1])
        return float(lo), float(hi)
    return float(s.min()), float(s.max())

# === 3) 6-panel summaries for first N features ===
N_FEATURES = 10  # change as needed

for feature in feature_cols[:N_FEATURES]:
    # shared limits for this feature across subplots
    x_min, x_max = feature_xlim(df[feature], trim_quantiles=(0.01, 0.99))

    fig = plt.figure(figsize=(14, 8))

    # (a) Histogram (two groups)
    ax1 = plt.subplot(2, 3, 1)
    sns.histplot(
        data=df, x=feature, hue="Label",
        kde=True, multiple="layer", element="step", stat="count",
        palette=palette_lbl, alpha=0.45, edgecolor=None
    )
    ax1.set_title(f"{feature} - Histogram")
    ax1.set_xlim(x_min, x_max)

    # (b) Boxplot
    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x="Label", y=feature, palette=palette_lbl)
    ax2.set_title(f"{feature} - Boxplot")
    ax2.set_ylim(x_min, x_max)

    # (c) Violin plot
    ax3 = plt.subplot(2, 3, 3)
    sns.violinplot(data=df, x="Label", y=feature, palette=palette_lbl, inner="box", cut=0)
    ax3.set_title(f"{feature} - Violin Plot")
    ax3.set_ylim(x_min, x_max)

    # (d) Strip plot
    ax4 = plt.subplot(2, 3, 4)
    sns.stripplot(data=df, x="Label", y=feature, palette=palette_lbl, jitter=True, alpha=0.65)
    ax4.set_title(f"{feature} - Strip Plot")
    ax4.set_ylim(x_min, x_max)

    # (e) KDE
    ax5 = plt.subplot(2, 3, 5)
    sns.kdeplot(data=df, x=feature, hue="Label", fill=True, common_norm=False,
                alpha=0.35, palette=palette_lbl)
    ax5.set_title(f"{feature} - KDE by Label")
    ax5.set_xlim(x_min, x_max)

    # (f) ECDF
    ax6 = plt.subplot(2, 3, 6)
    sns.ecdfplot(data=df, x=feature, hue="Label", palette=palette_lbl)
    ax6.set_title(f"{feature} - ECDF")
    ax6.set_xlim(x_min, x_max)

    # Panel letters (a)–(f)
    for i, ax in enumerate(fig.axes, start=1):
        ax.text(-0.12, 1.08, f"({chr(96+i)})", transform=ax.transAxes,
                size=12, weight="bold", va="bottom", ha="left")

    plt.tight_layout()
    out_path = f"plots1/miniplots/{feature}_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# === 4) PCA (2D & 3D) ===
X = df[feature_cols].copy()
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 2D
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)
pca_df_2d = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
pca_df_2d["Label"] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df_2d, x="PC1", y="PC2", hue="Label", palette=palette_lbl, alpha=0.8)
plt.title("PCA - 2D Projection")
plt.tight_layout()
plt.savefig("plots1/projections/pca_2d.png", dpi=300)
plt.close()

# PCA 3D
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
for label, color in palette_lbl.items():
    idx = (y == label).values
    ax.scatter(X_pca_3d[idx, 0], X_pca_3d[idx, 1], X_pca_3d[idx, 2],
               label=label, alpha=0.8, s=20, c=color)
ax.set_title("PCA - 3D Projection")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.savefig("plots1/projections/pca_3d.png", dpi=300)
plt.close()

# === 5) t-SNE 2D ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
tsne_df["Label"] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x="Dim1", y="Dim2", hue="Label", palette=palette_lbl, alpha=0.8)
plt.title("t-SNE - 2D Projection")
plt.tight_layout()
plt.savefig("plots1/projections/tsne_2d.png", dpi=300)
plt.close()

# === 6) Correlation heatmap ===
plt.figure(figsize=(12, 10))
corr = df[feature_cols].corr(numeric_only=True)
ax = sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True, square=True,
                 xticklabels=True, yticklabels=True)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("plots1/projections/correlation_heatmap.png", dpi=300)
plt.close()

# === 7) Pairplot (subset for clarity) ===
subset_cols = feature_cols[:5] + ["Label"]
sns.pairplot(df[subset_cols], hue="Label", palette=palette_lbl, diag_kind="kde")
plt.savefig("plots1/projections/pairplot.png", dpi=300)
plt.close()

print("✅ All visualizations saved under 'plots1/miniplots' and 'plots1/projections'.")

# %% Cell 31
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

# %% Cell 32
feature = "Signal Entropy"     # <-- change if needed
ORDER   = ["Healthy", "Parkinsonian"]
BLUE    = "#1f77b4"
ORANGE  = "#ff7f0e"
PALETTE = {"Healthy": BLUE, "Parkinsonian": ORANGE}

# optional: consistent x-limits with mild trimming
def xlim_trim(s, q=(0.01, 0.99)):
    s = s.dropna().astype(float)
    return float(s.quantile(q[0])), float(s.quantile(q[1]))

x_min, x_max = xlim_trim(df[feature], (0.01, 0.99))

fig = plt.figure(figsize=(14, 8))

# Histogram
plt.subplot(2, 3, 1)
sns.histplot(data=df, x=feature, hue="Label", element="step", stat="count", common_norm=False)
plt.title(f"{feature} - Histogram")
plt.ylim(0, df[feature].count() * 0.25)  # adjust the factor (0.25) as needed

# Boxplot
plt.subplot(2, 3, 2)
sns.boxplot(x='Label', y=feature, data=df)
plt.title(f"{feature} - Boxplot")
plt.ylim(df[feature].min() - 0.05, df[feature].max() + 0.05)  # pad y-range

# Violin plot
plt.subplot(2, 3, 3)
sns.violinplot(x='Label', y=feature, data=df)
plt.title(f"{feature} - Violin Plot")
plt.ylim(df[feature].min() - 0.05, df[feature].max() + 0.05)

# (d) Strip – guaranteed two colors (draw each group explicitly)
ax4 = plt.subplot(2, 3, 4)
for lab, color in zip(ORDER, [BLUE, ORANGE]):
    yvals = df.loc[df["Label"] == lab, feature]
    xvals = np.full(len(yvals), lab)
    sns.stripplot(x=xvals, y=yvals, color=color, jitter=0.25, alpha=0.6, zorder=2)
ax4.set_title(f"{feature} - Strip Plot")
ax4.set_xlabel("Label")

# set y-limits based on feature distribution
ymin, ymax = df[feature].min(), df[feature].max()
pad = (ymax - ymin) * 0.05  # 5% padding
ax4.set_ylim(ymin - pad, ymax + pad)


# (e) KDE – two colors via hue
ax5 = plt.subplot(2, 3, 5)
sns.kdeplot(data=df, x=feature, hue="Label", hue_order=ORDER,
            palette=PALETTE, fill=True, common_norm=False, alpha=0.35)
ax5.set_title(f"{feature} - KDE by Label")
ax5.set_xlim(x_min, x_max)

# (f) ECDF – two colors via hue
ax6 = plt.subplot(2, 3, 6)
sns.ecdfplot(data=df, x=feature, hue="Label", hue_order=ORDER, palette=PALETTE)
ax6.set_title(f"{feature} - ECDF")
ax6.set_xlim(x_min, x_max)

# consistent legends (same handles everywhere)
handles = [Patch(color=BLUE, label="Healthy"),
           Patch(color=ORANGE, label="Parkinsonian")]
for ax in (ax1, ax5, ax6):
    ax.legend(handles=handles, title="Label")

# panel letters
for i, ax in enumerate(fig.axes, start=1):
    ax.text(-0.12, 1.08, f"({chr(96+i)})", transform=ax.transAxes,
            size=12, weight="bold", va="bottom", ha="left")

plt.tight_layout()
plt.show()

# %% Cell 33
# === 1. Load real dataset ===
df = pd.read_csv("ctgan_true_positives_all1000.csv")  # Adjust path if needed

# === 2. Selected features for fuzzy sets ===
features_to_fuzzify = [
    "Signal Entropy", 
    "Smoothness", "Kurtosis"
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

# %% Cell 34
# === 1. Load your dataset ===
df = pd.read_csv("extracted_featuresnew.csv")  # Update this path if needed

# === 2. Define the updated fuzzy sets (triangular) ===
fuzzy_sets = {
    "Wave Energy": [
        ("Low", 1189.219723, 1925.035864, 2458.331934),
        ("Medium", 2280.812483, 2818.130214, 3901.356107),
        ("High", 3264.947691, 5148.650886, 14159.528815)
    ],
    "Irregularities": [
        ("Low", 141.0, 212.8, 249.0),
        ("Medium", 243.0, 271.0, 299.0),
        ("High", 287.6, 325.6, 510.0)
    ],
    "Signal Entropy": [
        ("Low", 2.142051, 2.239739, 2.265743),
        ("Medium", 2.253251, 2.276499, 2.288088),
        ("High", 2.284403, 2.292687, 2.301314)
    ],
    "Skewness": [
        ("Low", -0.479710, -0.164300, -0.036563),
        ("Medium", -0.069262, 0.022280, 0.085798),
        ("High", 0.068857, 0.187339, 0.402319)
    ],
    "Kurtosis": [
        ("Low", -1.321201, -1.252471, -1.170574),
        ("Medium", -1.186436, -1.131204, -1.070077),
        ("High", -1.094667, -0.984011, -0.563756)
    ],
    "Smoothness": [
        ("Low", 0.007047, 0.018559, 0.026212),
        ("Medium", 0.025296, 0.029371, 0.033433),
        ("High", 0.032110, 0.038441, 0.058630)
    ],
    "Average Amplitude": [
        ("Low", 0.007097, 0.015588, 0.025626),
        ("Medium", 0.023202, 0.030337, 0.037172),
        ("High", 0.035123, 0.045434, 0.063744)
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

# %% Cell 35
# === Triangular membership function ===
def triangular_membership(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# === Fuzzy sets for the 3 key features ===
fuzzy_sets_selected = {
    "Kurtosis": [
        ("Low", -1.321201, -1.252471, -1.170574),
        ("Medium", -1.186436, -1.131204, -1.070077),
        ("High", -1.094667, -0.984011, -0.563756)
    ],
    "Signal Entropy": [
        ("Low", 2.142051, 2.239739, 2.265743),
        ("Medium", 2.253251, 2.276499, 2.288088),
        ("High", 2.284403, 2.292687, 2.301314)
    ],
    "Smoothness": [
        ("Low", 0.007047, 0.018559, 0.026212),
        ("Medium", 0.025296, 0.029371, 0.033433),
        ("High", 0.032110, 0.038441, 0.058630)
    ]
}

# === Create one figure with 3 subplots ===
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharey=True)

for ax, (feature, sets) in zip(axes, fuzzy_sets_selected.items()):
    # Plot fuzzy sets
    for label, a, b, c in sets:
        x = np.linspace(a, c, 300)
        y = [triangular_membership(val, a, b, c) for val in x]
        ax.plot(x, y, label=f"{label} ({a:.3f}, {b:.3f}, {c:.3f})")
    
    # Overlay actual data distribution
    values = df[feature].dropna().values
    ax.scatter(values, np.zeros_like(values), alpha=0.3, color="black", s=15, label="Individuals")
    
    ax.set_title(f"Fuzzy Sets for {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fuzzy_sets_dataset1.png", dpi=300)  # Export as one image
plt.show()

print("✅ Saved: fuzzy_sets_dataset1.png")

# %% Cell 36
# === 1. Load the fuzzy-level CSV ===
df = pd.read_csv("individuals_with_updated_fuzzy_levels.csv")  # Update path if needed

# === 2. Define classification logic based on fuzzy labels ===
def classify(row):
    ampli = row["Average Amplitude Level"]
    entropy = row["Signal Entropy Level"]
    kurt = row["Kurtosis Level"]
    Irreg = row["Irregularities Level"]

    # Parkinsonian rule:
    if ((kurt in ["High", "Medium"]) and
        (entropy in ["Low", "Medium"]) and
        (ampli in ["Low", "Medium"])):
        return "Parkinsonian"

    # Healthy rule:
    elif ((ampli in ["High", "Medium"]) and
          (kurt == "Low") and
          (entropy == "High")):
        return "Healthy"

    # Not matching any rule
    else:
        return "None"

# === 3. Apply classification ===
df["Classification"] = df.apply(classify, axis=1)

# === 4. Save result ===
df.to_csv("classified_individuals_final.csv", index=False)
print("✅ Saved: classified_individuals_final.csv")

# %% Cell 37
# === 1. Load the classified dataset ===
df = pd.read_csv("classified_individuals_final.csv")  # Adjust path if needed

# === 2. Filter only the 'None' classifications ===
none_df = df[df["Classification"] == "None"]

# === 3. Save the filtered dataset ===
none_df.to_csv("none_classified_individuals.csv", index=False)
print("✅ Saved: none_classified_individuals.csv")

# %% Cell 38
# === 1. Load the "None"-classified dataset ===
df = pd.read_csv("none_classified_individuals.csv")

# === 2. Infer true label from filename ===
def infer_true_label(filename):
    if 'H' in filename:
        return "Healthy"
    elif 'P' in filename:
        return "Parkinsonian"
    else:
        return "Unknown"

df["True Label"] = df["Image"].apply(infer_true_label)

# === 3. Group by true label ===
healthy_df = df[df["True Label"] == "Healthy"]
parkinsonian_df = df[df["True Label"] == "Parkinsonian"]

# === 4. Count unique fuzzy-level patterns per group ===
features = ["Average Amplitude Level", "Signal Entropy Level", "Kurtosis Level"]

# Get most frequent patterns for Healthy
healthy_patterns = healthy_df[features].value_counts().reset_index(name="Count")
healthy_patterns["Group"] = "Healthy"

# Get most frequent patterns for Parkinsonian
parkinsonian_patterns = parkinsonian_df[features].value_counts().reset_index(name="Count")
parkinsonian_patterns["Group"] = "Parkinsonian"

# Combine and show
rule_candidates = pd.concat([healthy_patterns, parkinsonian_patterns], ignore_index=True)

# Sort for visibility
rule_candidates = rule_candidates.sort_values(by=["Group", "Count"], ascending=[True, False])

# === 5. Save the candidate rule patterns ===
rule_candidates.to_csv("candidate_fuzzy_rules_from_none.csv", index=False)
print("✅ Saved: candidate_fuzzy_rules_from_none.csv")

# %% Cell 39
# === 1. Load the full fuzzy-level dataset ===
df = pd.read_csv("classified_individuals_final.csv")  # Update path if needed

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
features = ["Smoothness Level", "Signal Entropy Level", "Kurtosis Level"]

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

# %% Cell 41
python -v

# %% Cell 42
# Jupyter shell/magic command removed from executable script: !python --version

# %% Cell 43
# === Load real and synthetic ===
df_real = pd.read_csv("extracted_featuresnew.csv")
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")

# === Match columns ===
feature_cols = [col for col in df_real.columns if col not in ["Image", "Label"]]

# === Ensure Label is same format ===
df_real["Origin"] = "Real"
df_synth["Origin"] = "Synthetic"
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# === Select only features with variation ===
valid_features = [
    col for col in feature_cols
    if df_real[col].nunique() > 1 and df_synth[col].nunique() > 1
]

# === 1. Multi-panel boxplots ===
ncols = 3
nrows = int(np.ceil(len(valid_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

for ax, col in zip(axes.flat, valid_features):
    sns.boxplot(x="Origin", y=col, data=df_combined, ax=ax)
    ax.set_title(f"Boxplot: {col}")

# Hide any unused subplots
for ax in axes.flat[len(valid_features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("boxplots_all_features.png", dpi=600)  # High-res
plt.close()

# === 2. Multi-panel histograms ===
nrows = int(np.ceil(len(valid_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

for ax, col in zip(axes.flat, valid_features):
    sns.histplot(df_real[col], color="blue", label="Real", kde=True, stat="density", ax=ax)
    sns.histplot(df_synth[col], color="orange", label="Synthetic", kde=True, stat="density", ax=ax)
    ax.set_title(f"Histogram: {col}")
    ax.legend()

for ax in axes.flat[len(valid_features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("histograms_all_features.png", dpi=600)  # High-res
plt.close()

# === 3. T-test comparison ===
for col in valid_features:
    try:
        t_stat, p_value = ttest_ind(df_real[col], df_synth[col], equal_var=False)
        result = "Same Distribution ✅" if p_value > 0.05 else "Different ⚠️"
        print(f"{col:25}: p = {p_value:.4f} → {result}")
    except Exception as e:
        print(f"{col:25}: Error — {e}")

# %% Cell 44
# ========================================
# DATASET 1 (WAVES) — THREE-STAGE STATISTICAL VALIDATION
# ========================================


# ========================================
# 1. LOAD DATA
# ========================================
# Real data (training + testing combined, or just testing)
df_real = pd.read_csv("extracted_featuresnewonlytesting.csv")

# Synthetic data from CTGAN (waves)
df_synth = pd.read_csv("ctgan_true_positives_all1000.csv")  # ή το ακριβές synthetic CSV για waves

# Add labels (if not present)
if 'Label' in df_real.columns:
    # If labels are strings, keep them; if numeric, convert
    pass
else:
    print("⚠ Warning: No 'Label' column found. Check your data.")

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
# 3. FEATURE COLUMNS — Dataset 1 (Waves)
# ========================================
# Drop identifier columns
exclude_cols = ["Image", "Label", "PatientID", "ID", "Unnamed: 0", "target", "Group"]
feature_cols = [col for col in df_real.columns if col not in exclude_cols]
print(f"\nFeatures to test: {feature_cols}\n")

# ========================================
# 4. THREE-STAGE VALIDATION FUNCTION
# ========================================
def three_stage_validation(real_data, synth_data, class_name, feature_cols):
    print("="*100)
    print(f"DATASET 1 (WAVES) — VALIDATION PROTOCOL — {class_name.upper()} CLASS")
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
            'Dataset': 'Dataset 1 (Waves)',
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

results_df.to_csv("statistical_validation_dataset1_waves.csv", index=False)

print("\n" + "="*100)
print("DATASET 1 (WAVES) — CONSOLIDATED RESULTS")
print("="*100)
print(results_df[['Class', 'Feature', 'Test_Used', 'Test_p_value', 'Wasserstein_Distance', 'Same_Distribution']].to_string(index=False))

n_total = len(all_results)
n_same = sum(r['Same_Distribution'] for r in all_results)

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"Total comparisons: {n_total}")
print(f"Same distribution: {n_same} ({100*n_same/n_total:.1f}%)")
print(f"\n✓ Results saved to: statistical_validation_dataset1_waves.csv")
