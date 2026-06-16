"""
Clean Python export from: NewSyntheticData 20250511.ipynb

All repeated import statements from the notebook have been consolidated here.
Notebook-only shell commands such as pip install/uninstall were commented out.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance, ttest_ind, shapiro, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.utils import resample


# %% Cell 0
# test ID == 0

# === 1. Define column names and input folder paths ===
column_names = ['X', 'Y', 'Z', 'Pressure', 'GripAngle', 'Timestamp', 'TestID']
parkinsonians_folder = r'C:\Users\adklt\OneDrive - aegean.gr\FinalPapers\PhDTestFolder\hw_dataset\hw_dataset\parkinson'
healthy_controls_folder = r'C:\Users\adklt\OneDrive - aegean.gr\FinalPapers\PhDTestFolder\hw_dataset\hw_dataset\control'

# === 2. Load and filter files for TestID == 0 ===
def load_and_combine_files(folder_path):
    combined_df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delimiter=';', header=None, names=column_names)
            df = df[df['TestID'] == 0]
            if not df.empty:
                df['PatientID'] = file_name
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# === 3. Segment data and compute features ===
def compute_features(data):
    segments = []
    current_segment = []
    for _, row in data.iterrows():
        if row['Pressure'] > 0:
            current_segment.append(row)
        elif current_segment:
            segments.append(pd.DataFrame(current_segment))
            current_segment = []
    if current_segment:
        segments.append(pd.DataFrame(current_segment))

    features = []
    for segment in segments:
        if len(segment) <= 1:
            continue
        x_diff = np.diff(segment['X'])
        y_diff = np.diff(segment['Y'])
        di = np.sum(np.sqrt(x_diff**2 + y_diff**2))
        Ti = len(segment) / 133  # 133 Hz sampling
        si = di / Ti if Ti > 0 else 0
        avg_pressure = segment['Pressure'].mean()

        features.append({
            'SegmentLength': di,
            'SegmentTime': Ti,
            'Speed': si,
            'AveragePressure': avg_pressure
        })

    if features:
        features_df = pd.DataFrame(features)
        return {
            'SegmentLength': features_df['SegmentLength'].mean(),
            'SegmentTime': features_df['SegmentTime'].mean(),
            'Speed': features_df['Speed'].mean(),
            'AveragePressure': features_df['AveragePressure'].mean()
        }
    else:
        return {
            'SegmentLength': 0,
            'SegmentTime': 0,
            'Speed': 0,
            'AveragePressure': 0
        }

# === 4. Compute descriptive statistics ===
def compute_detailed_statistics(stats_list):
    stats_df = pd.DataFrame(stats_list)
    detailed_stats = stats_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    detailed_stats['std'] = stats_df.std(numeric_only=True)
    return detailed_stats

# === 5. Load raw data ===
parkinsonians = load_and_combine_files(parkinsonians_folder)
healthy_controls = load_and_combine_files(healthy_controls_folder)

# === 6. Compute features and tag each row with PatientID ===
parkinsonian_stats = []
for patient_id in parkinsonians['PatientID'].unique():
    patient_data = parkinsonians[parkinsonians['PatientID'] == patient_id]
    features = compute_features(patient_data)
    features['PatientID'] = patient_id
    parkinsonian_stats.append(features)

healthy_stats = []
for patient_id in healthy_controls['PatientID'].unique():
    patient_data = healthy_controls[healthy_controls['PatientID'] == patient_id]
    features = compute_features(patient_data)
    features['PatientID'] = patient_id
    healthy_stats.append(features)

# === 7. Save real per-patient data ===
parkinsonian_df = pd.DataFrame(parkinsonian_stats)
parkinsonian_df['Group'] = 'Parkinsonian'
parkinsonian_df.to_csv('parkinsonian_real_data20250511.csv', index=False)

healthy_df = pd.DataFrame(healthy_stats)
healthy_df['Group'] = 'Healthy'
healthy_df.to_csv('healthy_real_data20250511.csv', index=False)

# === 8. Save statistical summaries ===
parkinsonian_detailed_stats = compute_detailed_statistics(parkinsonian_stats)
healthy_detailed_stats = compute_detailed_statistics(healthy_stats)

parkinsonian_detailed_stats.to_csv('parkinsonian_detailed_statistics20250511.csv', index=True)
healthy_detailed_stats.to_csv('healthy_detailed_statistics20250511.csv', index=True)

# === 9. Output ===
print("✅ Detailed per-patient data saved:")
print(" - parkinsonian_real_data20250511.csv")
print(" - healthy_real_data20250511.csv")

print("\n📊 Summary statistics saved:")
print(" - parkinsonian_detailed_statistics20250511.csv")
print(" - healthy_detailed_statistics20250511.csv")


# %% Cell 1
# ========== CONFIG ==========
latent_dim = 10
features = 4
n_samples = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_gen_size = 64  # how many samples to generate per loop during filtering

# ========== MODEL DEFINITIONS ==========

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, features)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(features, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ========== GAN TRAINING FUNCTION ==========

def train_gan(real_data, n_epochs=4000, batch_size=64):
    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.00005)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.00005)

    real_data = torch.tensor(real_data, dtype=torch.float32).to(device)

    for epoch in range(n_epochs):
        idx = np.random.randint(0, real_data.size(0), batch_size)
        real_batch = real_data[idx]
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_batch = G(z)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        D_loss = criterion(D(real_batch), real_labels) + criterion(D(fake_batch.detach()), fake_labels)
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        G_loss = criterion(D(G(z)), real_labels)
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    return G

# ========== VALIDATION FUNCTIONS ==========

def validate_synthetic_data(real_df, synthetic_df, class_name):
    print(f"\n🔍 Validation for {class_name} samples:")
    for feature in real_df.columns:
        w_dist = wasserstein_distance(real_df[feature], synthetic_df[feature])
        t_stat, p_value = ttest_ind(real_df[feature], synthetic_df[feature], equal_var=False)
        print(f"📌 {feature}")
        print(f"  Wasserstein Distance: {w_dist:.2f}")
        print(f"  T-test: t-stat = {t_stat:.2f}, p = {p_value:.4f}\n")

def plot_feature_comparison(real, synthetic, feature, title):
    plt.figure(figsize=(6, 4))
    plt.hist(real[feature], bins=30, alpha=0.6, label='Real', density=True)
    plt.hist(synthetic[feature], bins=30, alpha=0.6, label='Synthetic', density=True)
    plt.title(f'{title} - {feature}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========== MAIN PIPELINE ==========

for cls, file in [
    ('Parkinsonian', 'parkinsonian_real_data20250511.csv'),
    ('Healthy', 'healthy_real_data20250511.csv')
]:
    print(f"\n🚀 Training GAN for {cls}...")

    # Load real data
    real_df = pd.read_csv(file).drop(columns=['Group'], errors='ignore')
    feature_names = real_df.columns

    # Normalize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df)

    # Train GAN
    generator = train_gan(real_scaled)

    # Put generator in eval mode
    generator.eval()

    # Generate valid synthetic data in batches
    valid_samples = []
    attempts = 0
    required = n_samples
    features_to_check = ["SegmentLength", "SegmentTime"]
    feature_indices = [feature_names.get_loc(f) for f in features_to_check if f in feature_names]

    print(f"🔄 Generating {n_samples} valid synthetic samples for {cls}...")

    while len(valid_samples) < required:
        z = torch.randn(batch_gen_size, latent_dim).to(device)
        batch_scaled = generator(z).detach().cpu().numpy()
        batch_unscaled = scaler.inverse_transform(batch_scaled)

        for sample in batch_unscaled:
            if all(sample[i] >= 0 for i in feature_indices):
                valid_samples.append(sample)
                if len(valid_samples) >= required:
                    break

        attempts += batch_gen_size
        if attempts % 500 == 0:
            print(f"  ✅ {len(valid_samples)} valid / {attempts} attempts")

    synthetic_df = pd.DataFrame(valid_samples[:n_samples], columns=feature_names)

    # Save synthetic data and stats
    synthetic_df.to_csv(f'{cls.lower()}_synthetic_data20250511_optimized.csv', index=False)
    synthetic_df.describe().to_csv(f'{cls.lower()}_synthetic_statistics20250511_optimized.csv')

    print(f"✅ Synthetic data saved for {cls}.\n")

    # Validate
    validate_synthetic_data(real_df, synthetic_df, cls)

    # Optional: Plot
    for feature in feature_names:
        plot_feature_comparison(real_df, synthetic_df, feature, cls)


# %% Cell 2
# ---------- load your data ----------
df_real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").drop(columns=["Group"], errors="ignore")
df_synth_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv")

df_real_healthy   = pd.read_csv("healthy_real_data20250511.csv").drop(columns=["Group"], errors="ignore")
df_synth_healthy  = pd.read_csv("healthy_synthetic_data20250511_optimized.csv")

# ---------- helper: force numeric safely ----------
def coerce_numeric(df):
    df = df.copy()
    for c in df.columns:
        # skip obvious non-feature IDs if any slip in
        if c.lower() in {"image", "label", "origin", "group"}:
            continue
        # replace common non-numeric artifacts then coerce
        s = (df[c].astype(str)
                   .str.strip()
                   .str.replace(",", ".", regex=False))            # decimal commas -> dots
        s = s.str.replace(r"[^\d\.\-\+eE]", "", regex=True)          # remove units/symbols
        s = pd.to_numeric(s, errors="coerce")
        df[c] = s.replace([np.inf, -np.inf], np.nan)
    return df

df_real_parkinson  = coerce_numeric(df_real_parkinson)
df_synth_parkinson = coerce_numeric(df_synth_parkinson)
df_real_healthy    = coerce_numeric(df_real_healthy)
df_synth_healthy   = coerce_numeric(df_synth_healthy)

# ---------- choose only columns that are numeric in BOTH real & synthetic ----------
def numeric_feature_list(df_real, df_synth):
    common = [c for c in df_real.columns if c in df_synth.columns]
    numeric = []
    for c in common:
        if pd.api.types.is_numeric_dtype(df_real[c]) and pd.api.types.is_numeric_dtype(df_synth[c]):
            # keep if there is some variation after coercion
            if df_real[c].dropna().nunique() >= 2 and df_synth[c].dropna().nunique() >= 2:
                numeric.append(c)
    return numeric

# label origins for plotting
for df in (df_real_parkinson, df_synth_parkinson, df_real_healthy, df_synth_healthy):
    df["Origin"] = "Real" if df is df_real_parkinson or df is df_real_healthy else "Synthetic"

out_dir = "comparition_dataset3"
os.makedirs(out_dir, exist_ok=True)

def export_group(df_real, df_synth, group_name):
    cols = numeric_feature_list(df_real.drop(columns=["Origin"], errors="ignore"),
                                df_synth.drop(columns=["Origin"], errors="ignore"))
    if not cols:
        print(f"[{group_name}] No numeric features found after coercion.")
        return

    df_comb = pd.concat([df_real, df_synth], ignore_index=True)

    # 1) boxplots
    for col in cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Origin", y=col, data=df_comb, whis=1.5)
        plt.title(f"Boxplot: {col} ({group_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{group_name}_boxplot_{col}.png"), dpi=300)
        plt.close()

    # 2) histograms
    for col in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df_real[col].dropna(),  color="blue",   label="Real",      kde=True, stat="density")
        sns.histplot(df_synth[col].dropna(), color="orange", label="Synthetic", kde=True, stat="density")
        plt.title(f"Histogram: {col} ({group_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{group_name}_hist_{col}.png"), dpi=300)
        plt.close()

    # 3) stats table (t-test + Wasserstein)
    rows = []
    for col in cols:
        r = df_real[col].dropna()
        s = df_synth[col].dropna()
        try:
            t_stat, p_val = ttest_ind(r, s, equal_var=False)
            w_dist = wasserstein_distance(r, s)
            verdict = "Same Distribution ✅" if p_val > 0.05 else "Different ⚠️"
            rows.append([col, verdict, t_stat, p_val, w_dist, len(r), len(s)])
        except Exception as e:
            rows.append([col, f"Error — {e}", None, None, None, len(r), len(s)])

    pd.DataFrame(rows, columns=["Feature","Result","t_stat","p_value","Wasserstein","n_real","n_synth"]) \
      .to_csv(os.path.join(out_dir, f"{group_name}_validation.csv"), index=False)

    print(f"[{group_name}] Exported {len(cols)} features to {out_dir}/")

export_group(df_real_parkinson, df_synth_parkinson, "Parkinsonian")
export_group(df_real_healthy,   df_synth_healthy,   "Healthy")
print("✅ Done.")


# %% Cell 3
def plot_all_features(real_df, synthetic_df, class_name, save=False):
    features = real_df.columns
    fig, axes = plt.subplots(1, len(features), figsize=(18, 4))

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(real_df[feature], bins=30, color='blue', label='Real', kde=True, ax=ax, stat='density', element='step')
        sns.histplot(synthetic_df[feature], bins=30, color='orange', label='Synthetic', kde=True, ax=ax, stat='density', element='step')
        ax.set_title(f'{feature}')
        ax.legend()

    plt.tight_layout()
    plt.suptitle(f'{class_name} Feature Distributions (Real vs Synthetic)', fontsize=16, y=1.05)

    if save:
        plt.savefig(f'{class_name}_feature_distributions20250511.png', dpi=300, bbox_inches='tight')
    plt.show()


# %% Cell 4
plot_all_features(real_df, synthetic_df, cls, save=True)


# %% Cell 5
# Load real data
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Group="Healthy", Source="Real")
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Group="Parkinsonian", Source="Real")

# Load synthetic data
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Group="Healthy", Source="Synthetic")
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Group="Parkinsonian", Source="Synthetic")

# Combine into one DataFrame for each source
df_real = pd.concat([real_healthy, real_parkinson], ignore_index=True)
df_synthetic = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)


# %% Cell 6
# Load real data
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Group="Healthy", Source="Real")
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Group="Parkinsonian", Source="Real")

# Load synthetic data
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Group="Healthy", Source="Synthetic")
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Group="Parkinsonian", Source="Synthetic")

# Combine into one DataFrame for each source
df_real = pd.concat([real_healthy, real_parkinson], ignore_index=True)
df_synthetic = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)


# %% Cell 7
def plot_boxplots_by_group(df, source_label):
    features = [col for col in df.columns if col not in ['Group', 'Source']]
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Group', y=feature, data=df)
        plt.title(f'{feature} by Group ({source_label})')
        plt.tight_layout()
        plt.savefig(f'boxplot_{feature}_{source_label}.png', dpi=300)
        plt.show()


# %% Cell 8
plot_boxplots_by_group(df_real, "Real")
plot_boxplots_by_group(df_synthetic, "Synthetic")


# %% Cell 9
# Load real and synthetic datasets without labels
real_healthy = pd.read_csv("healthy_real_data20250511.csv").drop(columns=["Group"], errors='ignore')
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").drop(columns=["Group"], errors='ignore')
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv")
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv")

# Combine each
real_data = pd.concat([real_healthy, real_parkinson], ignore_index=True)
synthetic_data = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)


# %% Cell 10
def run_unsupervised_analysis(data, title=""):
    # Scale
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)

    # KMeans (just to check structure)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title(f"PCA + KMeans (k=2) on {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"PCA_KMeans_{title}.png", dpi=300)
    plt.show()

# Run separately
run_unsupervised_analysis(real_data, "Real Data")
run_unsupervised_analysis(synthetic_data, "Synthetic Data")


# %% Cell 11
# Assign dummy labels just for model testing SUpervised
real_data['Label'] = [0]*len(real_healthy) + [1]*len(real_parkinson)
synthetic_data['Label'] = [0]*len(synthetic_healthy) + [1]*len(synthetic_parkinson)


def run_classification(data, title=""):
    X = data.drop(columns=["Label"])
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"📊 Classification on {title}:\n")
    print(classification_report(y_test, y_pred))

# Optional: test classification
run_classification(real_data, "Real Data")
run_classification(synthetic_data, "Synthetic Data")


# %% Cell 12
# Notebook command removed/commented: !pip install shap


# %% Cell 13
# =======================
# 📥 1. Load Data
# =======================

# Load real and synthetic, with labels
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Label=0)
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Label=1)

synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)

# Combine
real_df = pd.concat([real_healthy, real_parkinson], ignore_index=True)
synthetic_df = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# Drop unnecessary columns
real_df = real_df.drop(columns=["Group"], errors="ignore")
feature_columns = ['SegmentLength', 'SegmentTime', 'Speed', 'AveragePressure']


# %% Cell 14
# Scale using same scaler for both
scaler = StandardScaler()
X_synthetic = scaler.fit_transform(synthetic_df[feature_columns])
y_synthetic = synthetic_df['Label']

X_real = scaler.transform(real_df[feature_columns])
y_real = real_df['Label']


# %% Cell 15
# Train on synthetic
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_synthetic, y_synthetic)

# Test on real
y_pred = model.predict(X_real)

print("📊 Classification Report (Trained on Synthetic → Tested on Real):\n")
print(classification_report(y_real, y_pred, digits=4))


# %% Cell 16
# ========================================
# COMPLETE STANDALONE SCRIPT — MULTIPLE TSTR RUNS
# ========================================


# ========================================
# 1. LOAD DATA
# ========================================
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Label=0)
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Label=1)

synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)

# Combine
real_df = pd.concat([real_healthy, real_parkinson], ignore_index=True)
synthetic_df = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# Drop unnecessary columns
real_df = real_df.drop(columns=["Group"], errors="ignore")
feature_columns = ['SegmentLength', 'SegmentTime', 'Speed', 'AveragePressure']

# ========================================
# 2. SCALE FEATURES
# ========================================
scaler = StandardScaler()
X_synthetic = scaler.fit_transform(synthetic_df[feature_columns])
y_synthetic = synthetic_df['Label']

X_real = scaler.transform(real_df[feature_columns])
y_real = real_df['Label']

print(f"Synthetic samples: {len(X_synthetic)}")
print(f"Real test samples: {len(X_real)}")
print(f"Features used: {feature_columns}\n")

# ========================================
# 3. MULTIPLE TSTR RUNS
# ========================================
N_RUNS = 50

f1_macro_scores = []
f1_class0_scores = []
f1_class1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []

print("="*80)
print(f"TSTR EVALUATION — {N_RUNS} RUNS")
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
# 4. AGGREGATE STATISTICS
# ========================================
print("\n" + "="*80)
print(f"AGGREGATE STATISTICS ({N_RUNS} RUNS)")
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
# 5. SAVE RESULTS
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
results_df.to_csv("tstr_multiple_runs.csv", index=False)
print(f"\n✓ Results saved to: tstr_multiple_runs.csv")

np.save("tstr_f1_macro_runs.npy", np.array(f1_macro_scores))
print(f"✓ F1 array saved to: tstr_f1_macro_runs.npy")


# %% Cell 17
# === 1. Load synthetic data (Healthy + Parkinsonian) ===
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)

df_synthetic = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# Balance synthetic dataset
df_h = df_synthetic[df_synthetic["Label"] == 0]
df_p = df_synthetic[df_synthetic["Label"] == 1]

df_h_res = resample(df_h, replace=True, n_samples=len(df_p), random_state=42)
df_balanced = pd.concat([df_h_res, df_p])

# Features/labels for training
X_synthetic = df_balanced.drop(columns=["Label", "Image"], errors="ignore").select_dtypes(include="number")
y_synthetic = df_balanced["Label"]

# === 2. Load real Dataset 3 ===
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Label=0)
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Label=1)

df_real3 = pd.concat([real_healthy, real_parkinson], ignore_index=True)
df_real3["target"] = df_real3["Label"]

X_real3 = df_real3.drop(columns=["Label", "Image", "target"], errors="ignore").select_dtypes(include="number")
y_real3 = df_real3["target"]

# Align features (same order as training)
X_real3 = X_real3[X_synthetic.columns]

# === 3. Train model with balanced class weights ===
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_synthetic, y_synthetic)

# Save model (optional)
joblib.dump(model, "random_forest_ctgan_dataset3.pkl")

# === 4. Predict and evaluate ===
y_pred3 = model.predict(X_real3)

print("\n📊 Classification Report (Dataset 3):")
print(classification_report(y_real3, y_pred3, digits=4))

print("\n🧩 Confusion Matrix (Dataset 3):")
print(confusion_matrix(y_real3, y_pred3))

print(f"\n✅ Accuracy: {accuracy_score(y_real3, y_pred3):.4f}")


# %% Cell 18
#K fold with out CTGAN

#  Dataset 3
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Label=0)
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Label=1)
df_real3 = pd.concat([real_healthy, real_parkinson], ignore_index=True)

df_real3["target"] = df_real3["Label"]
X = df_real3.drop(columns=["Label", "Image", "target"], errors="ignore").select_dtypes(include="number")
y = df_real3["target"]

print(f"Συνολικά real samples: {len(df_real3)}")
print(f"Healthy: {(y==0).sum()}, Parkinsonian: {(y==1).sum()}")

# K-Fold only real data
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

print(f"\nDataset 3 (Γραφίδα) — Real-only CV:")
print(f"F1 = {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"95% CI: [{np.percentile(f1_scores, 2.5):.3f}, {np.percentile(f1_scores, 97.5):.3f}]")
print(f"Min/Max: [{f1_scores.min():.3f}, {f1_scores.max():.3f}]")

np.save("rf_f1_dataset3_realonly.npy", f1_scores)


# %% Cell 21
# Get feature importance from the trained model
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display as a table
print("🔍 Feature Importances (from Random Forest):")
print(feature_importance_df)

# Plot
plt.figure(figsize=(6, 4))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=300)
plt.show()


# %% Cell 22
# Re-train model on synthetic data (just to ensure we're referencing the right one)
model_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
model_synthetic.fit(X_synthetic, y_synthetic)

# Extract importances
importances = model_synthetic.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display in console
print("🔍 Feature Importances (Trained on Synthetic Data):")
print(feature_importance_df)

# Plot
plt.figure(figsize=(6, 4))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='orange')
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest on Synthetic Data)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_synthetic_rf.png", dpi=300)
plt.show()


# %% Cell 23
# Re-train model on real data only
model_real = RandomForestClassifier(n_estimators=100, random_state=42)

# Scale real data
scaler_real = StandardScaler()
X_real_scaled = scaler_real.fit_transform(real_df[feature_columns])
y_real = real_df["Label"]

# Fit the model
model_real.fit(X_real_scaled, y_real)

# Extract feature importances
importances = model_real.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display
print("🔍 Feature Importances (Trained on Real Data):")
print(feature_importance_df)

# Plot
plt.figure(figsize=(6, 4))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest on Real Data)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_real_rf.png", dpi=300)
plt.show()


# %% Cell 24
# Load CSV (already in correct orientation)
stats_park = pd.read_csv("parkinsonian_synthetic_statistics20250511_optimized.csv", index_col=0)
stats_healthy = pd.read_csv("healthy_synthetic_statistics20250511_optimized.csv", index_col=0)

# Strip row index names just in case there are hidden spaces
stats_park.index = stats_park.index.str.strip()
stats_healthy.index = stats_healthy.index.str.strip()

def extract_fuzzy_sets(stats_df):
    fuzzy_sets = {}

    for feature in stats_df.columns:
        min_val = stats_df.at['min', feature]
        q1 = stats_df.at['25%', feature]
        q2 = stats_df.at['50%', feature]
        q3 = stats_df.at['75%', feature]
        max_val = stats_df.at['max', feature]

        fuzzy_sets[feature] = {
            'Low':    (min_val, q1, q2),
            'Medium': (q1, q2, q3),
            'High':   (q2, q3, max_val)
        }

    return fuzzy_sets

# Extract fuzzy sets
fuzzy_park = extract_fuzzy_sets(stats_park)
fuzzy_healthy = extract_fuzzy_sets(stats_healthy)

# Print fuzzy sets
def print_fuzzy_sets(fuzzy_sets, label):
    print(f"\n🔹 Fuzzy sets for {label}:")
    for feature, sets in fuzzy_sets.items():
        print(f"\n{feature}")
        for name, (a, b, c) in sets.items():
            print(f"  {name}: ({a:.2f}, {b:.2f}, {c:.2f})")

print_fuzzy_sets(fuzzy_park, "Parkinsonian Synthetic")
print_fuzzy_sets(fuzzy_healthy, "Healthy Synthetic")


# %% Cell 25
# Use your fuzzy sets already extracted
# fuzzy_park = extract_fuzzy_sets(stats_park)
# fuzzy_healthy = extract_fuzzy_sets(stats_healthy)

def triangle_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)

# Equal weights across all fuzzy levels and features
default_weights = {'Low': 0.33, 'Medium': 0.33, 'High': 0.33}

def compute_fuzzy_score(sample, fuzzy_set):
    score = 0
    for feature in fuzzy_set:
        val = sample[feature]
        memberships = {
            level: triangle_membership(val, *fuzzy_set[feature][level])
            for level in ['Low', 'Medium', 'High']
        }
        score += sum(memberships[level] * default_weights[level] for level in memberships)
    return score / len(fuzzy_set)  # Normalize to [0,1]

# Load real data
real_healthy = pd.read_csv("healthy_real_data20250511.csv").assign(Group="Healthy")
real_park = pd.read_csv("parkinsonian_real_data20250511.csv").assign(Group="Parkinsonian")
real_df = pd.concat([real_healthy, real_park], ignore_index=True)

# Score each real row
results = []
for idx, row in real_df.iterrows():
    park_score = compute_fuzzy_score(row, fuzzy_park)
    healthy_score = compute_fuzzy_score(row, fuzzy_healthy)
    label = "Parkinsonian" if park_score > healthy_score else "Healthy"
    results.append({
        "ParkinsonianScore": round(park_score, 3),
        "HealthyScore": round(healthy_score, 3),
        "LikelyClass": label
    })

# Combine with original real data
score_df = pd.concat([real_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
print(score_df[["ParkinsonianScore", "HealthyScore", "LikelyClass", "Group"]])


# %% Cell 26
# Map to 0 and 1 for sklearn
y_true = score_df['Group'].map({'Healthy': 0, 'Parkinsonian': 1}).values
y_pred = score_df['LikelyClass'].map({'Healthy': 0, 'Parkinsonian': 1}).values

print("\n📊 Fuzzy-Based Classification Report on Real Data:")
print(classification_report(y_true, y_pred, digits=4))

# Optional: just the F1 score
f1 = f1_score(y_true, y_pred)
print(f"✅ F1 Score: {f1:.4f}")


# %% Cell 27
# === 1. Load and merge synthetic datasets ===
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)
df = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# === 2. Select features to fuzzify ===
features_to_fuzzify = [
    "SegmentLength",
    "Speed",
    "AveragePressure"
]

# === 3. Function to compute tight triangular fuzzy sets ===
def tight_triangular_sets(values, feature):
    values = values.dropna()
    min_val = values.min()
    max_val = values.max()

    if min_val == max_val:
        return [{"Feature": feature, "FuzzySet": "All", "a": min_val, "b": min_val, "c": min_val}]

    p20 = np.percentile(values, 20)
    p35 = np.percentile(values, 35)
    p40 = np.percentile(values, 40)
    p50 = np.percentile(values, 50)
    p60 = np.percentile(values, 60)
    p65 = np.percentile(values, 65)
    p80 = np.percentile(values, 80)

    return [
        {"Feature": feature, "FuzzySet": "Low",    "a": min_val, "b": p20, "c": p40},
        {"Feature": feature, "FuzzySet": "Medium", "a": p35,     "b": p50, "c": p65},
        {"Feature": feature, "FuzzySet": "High",   "a": p60,     "b": p80, "c": max_val}
    ]

# === 4. Function to plot fuzzy sets ===
def plot_fuzzy_sets(fuzzy_defs, output_dir="fuzzy_plots"):
    os.makedirs(output_dir, exist_ok=True)
    features = set(fd["Feature"] for fd in fuzzy_defs)

    for feature in features:
        sets = [fd for fd in fuzzy_defs if fd["Feature"] == feature]
        xs = np.linspace(min(fd["a"] for fd in sets), max(fd["c"] for fd in sets), 500)

        plt.figure()
        for fd in sets:
            a, b, c = fd["a"], fd["b"], fd["c"]
            y = np.maximum(0, np.minimum((xs - a) / (b - a + 1e-9), (c - xs) / (c - b + 1e-9)))
            y[(xs <= a) | (xs >= c)] = 0
            y[xs == b] = 1
            plt.plot(xs, y, label=fd["FuzzySet"])

        plt.title(f"Fuzzy Sets for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature.replace(' ', '_')}_fuzzy_sets.png")
        plt.close()

# === 5. Generate fuzzy sets ===
fuzzy_definitions = []
for feature in features_to_fuzzify:
    if feature in df.columns:
        fuzzy_definitions.extend(tight_triangular_sets(df[feature], feature))
    else:
        print(f"⚠️ Feature not found in data: {feature}")

# === 6. Save to CSV ===
fuzzy_df = pd.DataFrame(fuzzy_definitions)
fuzzy_df.to_csv("tight_fuzzy_sets_from_synthetic.csv", index=False)
print("✅ Fuzzy sets saved to: tight_fuzzy_sets_from_synthetic.csv")

# === 7. Plot fuzzy sets ===
plot_fuzzy_sets(fuzzy_definitions)
print("📊 Fuzzy plots saved to: fuzzy_plots/")


# %% Cell 28
# === 1. Load and merge synthetic datasets ===
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)
df = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# === 2. Select features to fuzzify ===
features_to_fuzzify = [
    "SegmentLength",
    "Speed",
    "AveragePressure"
]

# === 3. Function to compute tight triangular fuzzy sets ===
def tight_triangular_sets(values, feature):
    values = values.dropna()
    min_val = values.min()
    max_val = values.max()

    if min_val == max_val:
        return [{"Feature": feature, "FuzzySet": "All", "a": min_val, "b": min_val, "c": min_val}]

    # percentiles for tight overlap
    p20 = np.percentile(values, 20)
    p35 = np.percentile(values, 35)
    p40 = np.percentile(values, 40)
    p50 = np.percentile(values, 50)
    p60 = np.percentile(values, 60)
    p65 = np.percentile(values, 65)
    p80 = np.percentile(values, 80)

    return [
        {"Feature": feature, "FuzzySet": "Low",    "a": min_val, "b": p20, "c": p40},
        {"Feature": feature, "FuzzySet": "Medium", "a": p35,     "b": p50, "c": p65},
        {"Feature": feature, "FuzzySet": "High",   "a": p60,     "b": p80, "c": max_val}
    ]

# === 4. Function to plot fuzzy sets (better visualization) ===
def plot_fuzzy_sets_better(df, fuzzy_defs, features, output_file="synthetic_fuzzy_sets.png"):
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 3*len(features)), sharey=True)

    if len(features) == 1:
        axes = [axes]  # ensure iterable

    for ax, feature in zip(axes, features):
        sets = [fd for fd in fuzzy_defs if fd["Feature"] == feature]
        xs = np.linspace(min(fd["a"] for fd in sets), max(fd["c"] for fd in sets), 500)

        # Plot fuzzy sets
        for fd in sets:
            a, b, c = fd["a"], fd["b"], fd["c"]
            y = np.maximum(0, np.minimum((xs - a) / (b - a + 1e-9), (c - xs) / (c - b + 1e-9)))
            y[(xs <= a) | (xs >= c)] = 0
            y[xs == b] = 1
            ax.plot(xs, y, linewidth=2, label=f"{fd['FuzzySet']} ({a:.2f}, {b:.2f}, {c:.2f})")

        # Overlay actual data distribution
        values = df[feature].dropna().values
        ax.scatter(values, np.zeros_like(values), alpha=0.25, color="black", s=12, label="Individuals")

        ax.set_title(f"Fuzzy Sets for {feature}", fontsize=12, weight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Membership Degree")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"✅ Combined fuzzy plot saved as: {output_file}")

# === 5. Generate fuzzy sets ===
fuzzy_definitions = []
for feature in features_to_fuzzify:
    if feature in df.columns:
        fuzzy_definitions.extend(tight_triangular_sets(df[feature], feature))
    else:
        print(f"⚠️ Feature not found in data: {feature}")

# === 6. Save fuzzy definitions ===
fuzzy_df = pd.DataFrame(fuzzy_definitions)
fuzzy_df.to_csv("tight_fuzzy_sets_from_synthetic.csv", index=False)
print("✅ Fuzzy sets saved to: tight_fuzzy_sets_from_synthetic.csv")

# === 7. Plot all selected features in one figure ===
plot_fuzzy_sets_better(df, fuzzy_definitions, features_to_fuzzify, output_file="synthetic_fuzzy_sets.png")


# %% Cell 29
# ---------------------------
# 1) DATA
# ---------------------------
# Synthetic (for deriving fuzzy sets + plotting individuals)
synthetic_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv").assign(Label=0)
synthetic_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv").assign(Label=1)
df_syn = pd.concat([synthetic_healthy, synthetic_parkinson], ignore_index=True)

# Stats (already computed per class)
stats_park = pd.read_csv("parkinsonian_synthetic_statistics20250511_optimized.csv", index_col=0)
stats_healthy = pd.read_csv("healthy_synthetic_statistics20250511_optimized.csv", index_col=0)
stats_park.index = stats_park.index.str.strip()
stats_healthy.index = stats_healthy.index.str.strip()

# Focus on these 3 features only
features = ["SegmentLength", "Speed", "AveragePressure"]

# ---------------------------
# 2) FUZZY SETS (tight triangles)
# ---------------------------
def tight_triangular_sets(values, feature):
    values = values.dropna()
    min_val, max_val = values.min(), values.max()
    if min_val == max_val:  # degenerate edge case
        return [{"Feature": feature, "FuzzySet": "All", "a": min_val, "b": min_val, "c": min_val}]
    p20 = np.percentile(values, 20)
    p35 = np.percentile(values, 35)
    p40 = np.percentile(values, 40)
    p50 = np.percentile(values, 50)
    p60 = np.percentile(values, 60)
    p65 = np.percentile(values, 65)
    p80 = np.percentile(values, 80)
    return [
        {"Feature": feature, "FuzzySet": "Low",    "a": min_val, "b": p20, "c": p40},
        {"Feature": feature, "FuzzySet": "Medium", "a": p35,     "b": p50, "c": p65},
        {"Feature": feature, "FuzzySet": "High",   "a": p60,     "b": p80, "c": max_val},
    ]

fuzzy_defs = []
for f in features:
    if f in df_syn.columns:
        fuzzy_defs.extend(tight_triangular_sets(df_syn[f], f))
    else:
        raise ValueError(f"Feature not found in synthetic data: {f}")

# ---------------------------
# 3) PLOT: one figure with 3 subplots
# ---------------------------
def triangular_membership(x, a, b, c):
    # safe triangular MF
    return np.maximum(0, np.minimum((x - a) / (max(b - a, 1e-12)),
                                    (c - x) / (max(c - b, 1e-12))))

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharey=True)

for ax, feature in zip(axes, features):
    sets = [fd for fd in fuzzy_defs if fd["Feature"] == feature]
    xs = np.linspace(min(fd["a"] for fd in sets), max(fd["c"] for fd in sets), 500)

    # Triangles
    for fd in sets:
        a, b, c, label = fd["a"], fd["b"], fd["c"], fd["FuzzySet"]
        y = triangular_membership(xs, a, b, c)
        ax.plot(xs, y, linewidth=2, label=f"{label} ({a:.2f}, {b:.2f}, {c:.2f})")

    # Overlay individuals (synthetic) on x-axis
    vals = df_syn[feature].dropna().values
    ax.scatter(vals, np.zeros_like(vals), s=12, alpha=0.25, color="black", label="Individuals")

    ax.set_title(f"Fuzzy Sets for {feature}", fontsize=12, weight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
os.makedirs("fuzzy_plots", exist_ok=True)
out_img = "fuzzy_plots/synthetic_fuzzy_sets_SegLen_Speed_AvgPress.png"
plt.savefig(out_img, dpi=300)
plt.show()
print(f"✅ Saved fuzzy figure: {out_img}")

# ---------------------------
# 4) STATS: only the 3 features, side-by-side (Healthy vs Parkinsonian)
# ---------------------------
wanted_rows = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

# subset & reorder rows safely
def safe_subset(df, rows, cols):
    rows = [r for r in rows if r in df.index]
    cols = [c for c in cols if c in df.columns]
    return df.loc[rows, cols]

park_sub = safe_subset(stats_park, wanted_rows, features)
healthy_sub = safe_subset(stats_healthy, wanted_rows, features)

# add a group column and stack for tidy view
park_sub["Group"] = "Parkinsonian"
healthy_sub["Group"] = "Healthy"
stats_tidy = pd.concat([healthy_sub, park_sub]).reset_index().rename(columns={"index": "Statistic"})

# pretty print to console
print("\n📊 Statistics for the 3 selected features (Synthetic):")
print(stats_tidy)

# save for your thesis appendix
stats_csv = "stats_synthetic_only3features.csv"
stats_tidy.to_csv(stats_csv, index=False)
print(f"✅ Saved stats table: {stats_csv}")


# %% Cell 30
# === 1. Load your data ===
df_healthy = pd.read_csv("healthy_real_data20250511.csv")
df_parkinsonian = pd.read_csv("parkinsonian_real_data20250511.csv")
fuzzy_sets = pd.read_csv("tight_fuzzy_sets_from_synthetic.csv")

# === 2. Combine both datasets ===
df = pd.concat([df_healthy, df_parkinsonian], ignore_index=True)

# === 3. Fuzzy membership function (triangular) ===
def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a + 1e-9)
    elif b < x < c:
        return (c - x) / (c - b + 1e-9)
    return 0

# === 4. Apply fuzzy categorization ===
fuzzy_results = []

features = fuzzy_sets['Feature'].unique()

for _, row in df.iterrows():
    result = {'PatientID': row['PatientID'], 'Group': row['Group']}
    for feature in features:
        value = row[feature]
        # Get fuzzy sets for this feature
        sets = fuzzy_sets[fuzzy_sets['Feature'] == feature]
        memberships = {}
        for _, fuzzy in sets.iterrows():
            label = fuzzy['FuzzySet']
            a, b, c = fuzzy['a'], fuzzy['b'], fuzzy['c']
            memberships[label] = triangular_membership(value, a, b, c)
        # Assign the label with highest membership
        best_label = max(memberships, key=memberships.get)
        result[f"{feature}_Category"] = best_label
    fuzzy_results.append(result)

# === 5. Save fuzzy labels per patient ===
fuzzy_df = pd.DataFrame(fuzzy_results)
fuzzy_df.to_csv("fuzzy_categorized_patients20250511.csv", index=False)

print("✅ Fuzzy categories assigned and saved to: fuzzy_categorized_patients20250511.csv")


# %% Cell 31
# === 1. Load the fuzzy categorized data ===
fuzzy_df = pd.read_csv("fuzzy_categorized_patients20250511.csv")

# === 2. Create a string-based rule signature for each patient ===
fuzzy_df['Rule'] = (
    "SegmentLength=" + fuzzy_df['SegmentLength_Category'] + " | " +
    "Speed=" + fuzzy_df['Speed_Category'] + " | " +
    "AveragePressure=" + fuzzy_df['AveragePressure_Category']
)

# === 3. Count rule occurrences by Group ===
rule_counts = fuzzy_df.groupby(['Group', 'Rule']).size().reset_index(name='Count')

# === 4. Pivot for easier comparison (optional) ===
pivot = rule_counts.pivot(index='Rule', columns='Group', values='Count').fillna(0).astype(int)
pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

# === 5. Save results ===
pivot.to_csv("fuzzy_rule_counts_by_group20250511.csv")
print("✅ Saved fuzzy rule occurrence counts to: fuzzy_rule_counts_by_group20250511.csv")

# Optional: print top rules
print("\n📊 Top fuzzy rules by group:")
print(pivot.head(10))


# %% Cell 32
# Notebook command removed/commented: !pip uninstall pandas -y
# Notebook command removed/commented: !pip uninstall pandas -y
# Notebook command removed/commented: !pip uninstall pandas -y


# %% Cell 33
# Notebook command removed/commented: !pip uninstall numpy -y
# Notebook command removed/commented: !pip uninstall numpy -y
# Notebook command removed/commented: !pip uninstall scipy -y
# Notebook command removed/commented: !pip uninstall scipy -y


# %% Cell 34
# Notebook command removed/commented: !pip uninstall numpy scipy pandas scikit-learn matplotlib seaborn shap sdv -y
# Notebook command removed/commented: !pip uninstall numpy scipy pandas scikit-learn matplotlib seaborn shap sdv -y
# Notebook command removed/commented: !pip uninstall numpy scipy pandas scikit-learn matplotlib seaborn shap sdv -y


# %% Cell 35
# Notebook command removed/commented: pip install "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" "scikit-learn==1.5.0"


# %% Cell 36
# ========================================
# 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# ========================================

# Synthetic data per class
synth_healthy = pd.read_csv("healthy_synthetic_data20250511_optimized.csv")
synth_parkinson = pd.read_csv("parkinsonian_synthetic_data20250511_optimized.csv")

# Real data per class
real_healthy = pd.read_csv("healthy_real_data20250511.csv")
real_parkinson = pd.read_csv("parkinsonian_real_data20250511.csv")

print(f"Real Healthy:           {len(real_healthy)} samples")
print(f"Real Parkinsonian:      {len(real_parkinson)} samples")
print(f"Synthetic Healthy:      {len(synth_healthy)} samples")
print(f"Synthetic Parkinsonian: {len(synth_parkinson)} samples")

# ========================================
# 2. FEATURE COLUMNS
# ========================================
# Drop identifier columns if they exist
exclude_cols = ["Image", "Label", "PatientID", "ID", "Unnamed: 0"]
feature_cols = [col for col in real_healthy.columns if col not in exclude_cols]
print(f"\nFeatures to test: {feature_cols}\n")

# ========================================
# 3. THREE-STAGE VALIDATION FUNCTION
# ========================================

def three_stage_validation(real_data, synth_data, class_name, feature_cols):
    """
    Three-stage statistical validation:
    Stage 1: Shapiro-Wilk normality
    Stage 2: t-test (if normal) or Mann-Whitney U (if not)
    Stage 2b: Wasserstein distance (always)
    Stage 3: Combined interpretation
    """
    print("="*100)
    print(f"VALIDATION PROTOCOL — {class_name.upper()} CLASS")
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

        # STAGE 1: Shapiro-Wilk Normality
        # Note: shapiro accepts max 5000 samples, sample if needed
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

        # STAGE 2b: Wasserstein Distance
        wd = wasserstein_distance(real, synth)

        # STAGE 3: Combined Interpretation
        same_dist = test_p > 0.05

        # Print formatted output
        print(f"\n📌 {col}")
        print(f"   Shapiro-Wilk Real:       p = {sw_real_p:.4f} → {'Normal ✓' if real_normal else 'NOT Normal ⚠'}")
        print(f"   Shapiro-Wilk Synthetic:  p = {sw_synth_p:.4f} → {'Normal ✓' if synth_normal else 'NOT Normal ⚠'}")
        print(f"   ➔ {'Both normal' if both_normal else 'At least one non-normal'} → using {test_name}")
        print(f"   {test_name}:           {test_label}-stat = {test_stat:.4f}, p = {test_p:.4f}")
        print(f"   Wasserstein Distance:    {wd:.4f}")
        print(f"   FINAL: {'Same Distribution ✓' if same_dist else 'Different ⚠'}")

        results.append({
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
# 4. RUN VALIDATION FOR BOTH CLASSES
# ========================================

# Healthy class
healthy_results = three_stage_validation(
    real_data=real_healthy,
    synth_data=synth_healthy,
    class_name="Healthy",
    feature_cols=feature_cols
)

print("\n")

# Parkinsonian class
parkinson_results = three_stage_validation(
    real_data=real_parkinson,
    synth_data=synth_parkinson,
    class_name="Parkinsonian",
    feature_cols=feature_cols
)

# ========================================
# 5. CONSOLIDATED RESULTS
# ========================================
all_results = healthy_results + parkinson_results
results_df = pd.DataFrame(all_results)

# Save full results
results_df.to_csv("statistical_validation_full_results.csv", index=False)
print("\n" + "="*100)
print("CONSOLIDATED RESULTS")
print("="*100)
print(results_df.to_string(index=False))

# Summary
n_total = len(all_results)
n_same = sum(r['Same_Distribution'] for r in all_results)

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"Total comparisons: {n_total}")
print(f"Same distribution: {n_same} ({100*n_same/n_total:.1f}%)")
print(f"Different:         {n_total - n_same} ({100*(n_total-n_same)/n_total:.1f}%)")
print(f"\n✓ Results saved to: statistical_validation_full_results.csv")

# Per-class summary
print(f"\n--- Healthy class ---")
healthy_same = sum(r['Same_Distribution'] for r in healthy_results)
print(f"  {healthy_same}/{len(healthy_results)} features show equivalent distributions")

print(f"\n--- Parkinsonian class ---")
park_same = sum(r['Same_Distribution'] for r in parkinson_results)
print(f"  {park_same}/{len(parkinson_results)} features show equivalent distributions")
