import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# KONFIGURASI TERMINAL WINDOWS & TAMPILAN TABEL POLARS
# ==============================================================================
sys.stdout.reconfigure(encoding='utf-8')
pl.Config.set_tbl_formatting("ASCII_FULL")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    classification_report,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Pengaturan Global Visualisasi Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# MAPPING NAMA SUB-SEKTOR UNTUK PENJELASAN HEATMAP YANG RAPI
# ==============================================================================
LABEL_MAP = {
    'Mining and quarrying B': 'Mining (B)',
    'Manufacture of food products, beverages and tobacco products CA': 'Food & Tobacco (CA)',
    'Manufacture of textiles, apparel, leather and related products CB': 'Textiles & Leather (CB)',
    'Manufacture of wood and paper products, and printing CC': 'Wood & Paper (CC)',
    'Manufacture of coke, and refined petroleum products CD': 'Coke & Petroleum (CD)',
    'Manufacture of chemicals and chemical products CE': 'Chemicals (CE)',
    'Manufacture of pharma-ceuticals, medicinal chemical and botanical products CF': 'Pharma (CF)',
    'Manufacture of rubber and plastics products, and other non-metallic mineral products CG': 'Rubber & Plastics (CG)',
    'Manufacture of basic metals and fabricated metal products, except machinery and equipment CH': 'Basic Metals (CH)',
    'Manufacture of computer, electronic and optical products CI': 'Electronics (CI)',
    'Manufacture of electrical equipment CJ': 'Electrical Eq. (CJ)',
    'Manufacture of machinery and equipment n.e.c. CK': 'Machinery (CK)',
    'Manufacture of transport equipment CL': 'Transport Eq. (CL)',
    'Other manufacturing, and repair and installation of machinery and equipment CM': 'Other Mfg. (CM)',
    'Manufacturing C': 'Manufacturing (C)',
    'Electricity, gas, steam and air conditioning supply D': 'Electricity & Gas (D)',
    'Industry (excluding water and waste management) B+C+D': 'Industry (excl. water) B+C+D'
}

# ==============================================================================
# 0. MEMBACA & PREPROCESSING DATA (POLARS)
# ==============================================================================
file_path = "stadat-ipa0037-13.2.1.7-en.xlsx"

try:
    df_raw = pl.read_excel(file_path, sheet_id=1, has_header=False, engine="calamine")
except Exception:
    df_raw = pl.read_excel(file_path, sheet_id=1, has_header=False)

# Ambil header dari Baris Index 1
raw_headers = df_raw.row(1)

# Penanganan nama kolom unik
header_cols = []
seen = {}
for val in raw_headers:
    name = str(val).strip() if val is not None else "Unnamed"
    if name in seen:
        seen[name] += 1
        name = f"{name}_{seen[name]}"
    else:
        seen[name] = 0
    header_cols.append(name)

# Slice data mulai baris index 3 (Data numerik)
df_polars = df_raw.slice(3)
df_polars.columns = header_cols

# Forward fill kolom Year
df_polars = df_polars.with_columns(pl.col("Year").forward_fill())

# Fitur numerik awal
all_feature_cols = [c for c in header_cols if c not in ["Year", "Period Code"] and not c.startswith("Unnamed")]

# Cast tipe data ke Float64 & bersihkan null
exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in all_feature_cols]
df_polars = df_polars.with_columns(exprs).drop_nulls(subset=all_feature_cols)

print(f"Data Berhasil Dimuat! Total Dataset: {df_polars.shape[0]} Baris x {df_polars.shape[1]} Kolom")

# ==============================================================================
# 1. FEATURE SELECTION VÍA CORRELATION THRESHOLD (|r| >= 0.4) & ANNOTATED HEATMAP
# ==============================================================================
print("\n--- 1. Melakukan Seleksi Variabel dengan Correlation Threshold (|r| >= 0.4) ---")
X_all = df_polars.select(all_feature_cols).to_numpy()
corr_full = np.corrcoef(X_all, rowvar=False)

# Target referensi agregat: Industry (excluding water and waste management) B+C+D
target_idx = all_feature_cols.index('Industry (excluding water and waste management) B+C+D')

# Pilih variabel yang memiliki korelasi absolut >= 0.4
selected_indices = [i for i in range(len(all_feature_cols)) if abs(corr_full[i, target_idx]) >= 0.4]
feature_cols = [all_feature_cols[i] for i in selected_indices]
short_labels = [LABEL_MAP.get(c, c) for c in feature_cols]

print(f"Variabel Terpilih (Threshold |r| >= 0.4): {len(feature_cols)} dari {len(all_feature_cols)} Variabel")

# Hitung korelasi untuk variabel terpilih
X_raw = df_polars.select(feature_cols).to_numpy()
corr_matrix = np.corrcoef(X_raw, rowvar=False)

# Visualisasi Heatmap (Ukuran Ringkas & Dilengkapi Angka di Dalam Kotak)
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Tambahkan Angka Korelasi di Setiap Sel Kotak Heatmap
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix[i, j]
        text_color = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5, fontweight="bold")

# Colorbar & Sumbu
cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
cbar.set_label('Koefisien Korelasi (r)', fontsize=8.5, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

ax.set_xticks(np.arange(len(feature_cols)))
ax.set_yticks(np.arange(len(feature_cols)))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7.5, fontweight='semibold')
ax.set_yticklabels(short_labels, fontsize=7.5, fontweight='semibold')
ax.set_title("Correlation Heatmap (Filtered |r| ≥ 0.4)", fontsize=11, fontweight='bold', pad=12)

plt.tight_layout()
plt.show()

# Standardisasi Data Terpilih
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ==============================================================================
# 2. CONTINUAL EVALUATION METRICS (4 METHOD RINGKAS: ELBOW, SILHOUETTE, DB, CHI)
# ==============================================================================
print("\n--- 2. Pengujian 4 Metode Evaluasi Klaster (k = 2 hingga 6) ---")
k_range = range(2, 7)
metrics_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    metrics_list.append({
        "k": k, 
        "WCSS": km.inertia_, 
        "Silhouette": silhouette_score(X_scaled, labels), 
        "Davies_Bouldin": davies_bouldin_score(X_scaled, labels), 
        "Calinski_Harabasz": calinski_harabasz_score(X_scaled, labels)
    })

df_metrics = pl.DataFrame(metrics_list)
print(df_metrics)

# Grid 2x2 Evaluasi Metrik (Ukuran Ringkas: 7.5 x 5.5)
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), dpi=120)
fig.subplots_adjust(hspace=0.38, wspace=0.32)

plot_configs = [
    (axes[0, 0], "WCSS", "Elbow Method (WCSS)", "#1f77b4", 'o'),
    (axes[0, 1], "Silhouette", "Silhouette Score (Tinggi)", "#2ca02c", 's'),
    (axes[1, 0], "Davies_Bouldin", "Davies-Bouldin Index (Rendah)", "#d62728", '^'),
    (axes[1, 1], "Calinski_Harabasz", "Calinski-Harabasz Index (Tinggi)", "#9467bd", 'd')
]

for ax, col, title, color, marker in plot_configs:
    ax.plot(df_metrics["k"], df_metrics[col], marker=marker, color=color, linewidth=1.5, markersize=5)
    
    # Highlight nilai optimal k=3
    val_k3 = df_metrics.filter(pl.col("k") == 3)[col][0]
    ax.scatter([3], [val_k3], color='gold', s=70, zorder=5, edgecolors='black', label='Selected (k=3)')
    
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=5)
    ax.set_xlabel("Jumlah Cluster (k)", fontsize=7.5)
    ax.set_xticks(list(k_range))
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=7)

plt.tight_layout()
plt.show()

# ==============================================================================
# 3. K-MEANS CLUSTERING (k=3) & PENAMBAHAN LABEL KE DATAFRAME
# ==============================================================================
print("\n--- 3. Menjalankan K-Means Clustering (k=3) & Penambahan Label Hasil ---")
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_3.fit_predict(X_scaled)

# Menambahkan kolom label klaster ke Polars DataFrame
df_polars = df_polars.with_columns(pl.Series("Cluster_Label", cluster_labels))

print("Distribusi Anggota Tiap Cluster (k=3):")
print(df_polars.group_by("Cluster_Label").len().sort("Cluster_Label"))

# Visualisasi PCA 2D Cluster
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans_3.cluster_centers_)

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=120)
colors = ['#e41a1c', '#377eb8', '#4daf4a']

for c in range(3):
    mask = (cluster_labels == c)
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1], 
        c=colors[c], label=f'Cluster {c}', 
        s=60, alpha=0.85, edgecolors='white', linewidth=0.5
    )

ax.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1], 
    c='black', marker='X', s=140, label='Centroid', zorder=6
)

ax.set_title("Visualisasi 3 Cluster K-Means via PCA 2D", fontsize=10.5, fontweight='bold', pad=10)
ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=8.5)
ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=8.5)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. DATA SPLITTING & MACHINE LEARNING (RANDOM FOREST)
# ==============================================================================
print("\n--- 4. Data Splitting (Training 70%, Validation 15%, Testing 15%) ---")
X = X_scaled
y = df_polars["Cluster_Label"].to_numpy()

# Split 1: Pisahkan 15% untuk Test Set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Split 2: Pisahkan Training (70%) dan Validation (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"Ukuran Dataset -> Training: {len(X_train)} | Validation: {len(X_val)} | Testing: {len(X_test)}")

print("\n--- 5. Pelatihan Machine Learning (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi Validation Set
val_preds = rf_model.predict(X_val)
print(f"\nAkurasi Validation Set: {accuracy_score(y_val, val_preds) * 100:.2f}%")
print("[ Classification Report Validation Set ]")
print(classification_report(y_val, val_preds))

# Evaluasi Test Set
test_preds = rf_model.predict(X_test)
print(f"\nAkurasi Test Set: {accuracy_score(y_test, test_preds) * 100:.2f}%")
print("[ Classification Report Test Set ]")
print(classification_report(y_test, test_preds))

# Feature Importance Bar Chart
df_importance = pl.DataFrame({
    "Feature": [LABEL_MAP.get(c, c) for c in feature_cols],
    "Importance": rf_model.feature_importances_
}).sort("Importance", descending=True)

fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
bars = ax.barh(df_importance["Feature"].to_list()[::-1], df_importance["Importance"].to_list()[::-1], color='#008080', height=0.6)
ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=7.5, fontweight='bold')

ax.set_xlim(0, max(df_importance["Importance"]) * 1.15)
ax.set_title("Feature Importance (Random Forest)", fontsize=10.5, fontweight='bold', pad=10)
ax.set_xlabel("Importance Score", fontsize=8.5)
ax.tick_params(labelsize=8)
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()