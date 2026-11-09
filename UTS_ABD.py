import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

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

# ==============================================================================
# KONFIGURASI GLOBAL & TAMPILAN (SYSTEM & MATPLOTLIB)
# ==============================================================================
# 1. Mencegah UnicodeEncodeError di Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')
pl.Config.set_tbl_formatting("ASCII_FULL")

# 2. Pengaturan Presisi Visualisasi Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# 3. Dictionary Mapping Nama Sub-Sektor Industri (Agar Grafik Rapi & Terbaca)
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
# TAHAP 1: PREPROCESSING DATA & IMPUTASI (POLARS)
# ==============================================================================
print("=== TAHAP 1: PREPROCESSING DATA & IMPUTASI ===")
file_path = "stadat-ipa0037-13.2.1.7-en.xlsx"

# 1. Baca Sheet Excel
try:
    df_raw = pl.read_excel(file_path, sheet_id=1, has_header=False, engine="calamine")
except Exception:
    df_raw = pl.read_excel(file_path, sheet_id=1, has_header=False)

# 2. Ambil & Format Nama Kolom dari Baris Index 1 (Baris ke-2 Excel)
raw_headers = df_raw.row(1)
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

# 3. Slicing Data Numerik Utama & Forward Fill Kolom 'Year'
df_polars = df_raw.slice(3)
df_polars.columns = header_cols
df_polars = df_polars.with_columns(pl.col("Year").forward_fill())

# 4. Filter Fitur Numerik & Pembersihan Nilai Kosong (Null)
all_feature_cols = [c for c in header_cols if c not in ["Year", "Period Code"] and not c.startswith("Unnamed")]
exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in all_feature_cols]
df_polars = df_polars.with_columns(exprs).drop_nulls(subset=all_feature_cols)

print(f"[OK] Data Preprocessing Selesai. Dimensi Dataset: {df_polars.shape[0]} Baris x {df_polars.shape[1]} Kolom\n")


# ==============================================================================
# TAHAP 2: SELEKSI FITUR (CORRELATION THRESHOLD |r| >= 0.4) & HEATMAP
# ==============================================================================
print("=== TAHAP 2: SELEKSI FITUR & CORRELATION HEATMAP ===")

X_all = df_polars.select(all_feature_cols).to_numpy()
corr_full = np.corrcoef(X_all, rowvar=False)

# Target Referensi: Industry (excluding water and waste management) B+C+D
target_idx = all_feature_cols.index('Industry (excluding water and waste management) B+C+D')

# Saring hanya variabel dengan korelasi absolut |r| >= 0.4 terhadap target
selected_indices = [i for i in range(len(all_feature_cols)) if abs(corr_full[i, target_idx]) >= 0.4]
feature_cols = [all_feature_cols[i] for i in selected_indices]
short_labels = [LABEL_MAP.get(c, c) for c in feature_cols]

print(f"[OK] Seleksi Fitur Selesai: {len(feature_cols)} Variabel Terpilih dari {len(all_feature_cols)} Variabel Awal")

# Matriks Korelasi Terpilih
X_raw = df_polars.select(feature_cols).to_numpy()
corr_matrix = np.corrcoef(X_raw, rowvar=False)

# Visualisasi Correlation Heatmap Presisi (Dengan Angka Nilai Korelasi)
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Tampilkan angka presisi di setiap sel
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix[i, j]
        text_color = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5, fontweight="bold")

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

# Standardisasi Z-Score pada Data Terpilih
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print("[OK] Standardisasi Z-Score Selesai.\n")


# ==============================================================================
# TAHAP 3: EVALUASI KLASTER CONTINUAL (4 METODE: ELBOW, SILHOUETTE, DB, CHI)
# ==============================================================================
print("=== TAHAP 3: EVALUASI METRIK KLASTER (k = 2 - 6) ===")
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

# Visualisasi Grid 2x2 Evaluasi Metrik Ringkas
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
    
    # Highlight nilai k=3
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
print("[OK] Evaluasi 4 Metrik Selesai.\n")


# ==============================================================================
# TAHAP 4: K-MEANS CLUSTERING (k=3) & INJEKSI LABEL BARU
# ==============================================================================
print("=== TAHAP 4: K-MEANS CLUSTERING (k=3) & PENAMBAHAN LABEL ===")
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_3.fit_predict(X_scaled)

# Penambahan Kolom Label Hasil Klaster Terakhir ke Dalam Polars DataFrame
df_polars = df_polars.with_columns(pl.Series("Cluster_Label", cluster_labels))

print("Distribusi Anggota Tiap Cluster (k=3):")
print(df_polars.group_by("Cluster_Label").len().sort("Cluster_Label"))

# Visualisasi Sebaran 2D Klaster via PCA
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
print("[OK] K-Means Clustering & Injeksi Label Selesai.\n")


# ==============================================================================
# TAHAP 5: DATA SPLITTING (TRAIN 70%, VALIDATION 15%, TESTING 15%)
# ==============================================================================
print("=== TAHAP 5: DATA SPLITTING (TRAIN, VALIDATION, TEST) ===")
X = X_scaled
y = df_polars["Cluster_Label"].to_numpy()

# 1. Split Pertama: Pisahkan Test Set (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2. Split Kedua: Membagi Sisa Data Menjadi Train (70%) dan Validation (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"[OK] Ukuran Dataset -> Training: {len(X_train)} | Validation: {len(X_val)} | Testing: {len(X_test)}\n")


# ==============================================================================
# TAHAP 6: MACHINE LEARNING (RANDOM FOREST) & EVALUASI MODEL
# ==============================================================================
print("=== TAHAP 6: MACHINE LEARNING (RANDOM FOREST CLASSIFIER) ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 1. Evaluasi pada Validation Set (15%)
val_preds = rf_model.predict(X_val)
print(f"Akurasi Validation Set: {accuracy_score(y_val, val_preds) * 100:.2f}%")
print("[ Classification Report Validation Set ]")
print(classification_report(y_val, val_preds))

# 2. Evaluasi pada Test Set (15%)
test_preds = rf_model.predict(X_test)
print(f"Akurasi Test Set: {accuracy_score(y_test, test_preds) * 100:.2f}%")
print("[ Classification Report Test Set ]")
print(classification_report(y_test, test_preds))

# 3. Visualisasi Top Feature Importance
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
print("=== SELURUH PROSES SELESAI ===")
