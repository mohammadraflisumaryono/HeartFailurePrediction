# %% [markdown]
# # Prediksi Penyakit Jantung (Heart Disease Prediction)
# **Dataset**: Heart Failure Prediction (918 sampel)  
# **Jenis**: Klasifikasi biner  
# **Target**: HeartDisease (0 = sehat, 1 = sakit)
# 

# %% [markdown]
# ## 1. Import Library
# Pada tahap ini, mengimpor library yang diperlukan untuk analisis data, visualisasi, preprocessing, dan pemodelan.  
# - `pandas` dan `numpy`: Untuk manipulasi dan perhitungan data.  
# - `matplotlib` dan `seaborn`: Untuk visualisasi data.  
# - `sklearn`: Untuk preprocessing, pembagian data, pemodelan, dan evaluasi.  
# - `warnings`: Mengabaikan peringatan yang tidak relevan.
# 

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os


# %% [markdown]
#  ## 2. Load Data
#  Dataset `heart.csv` dimuat menggunakan `pandas` untuk analisis.  
#  - Menampilkan bentuk dataset (jumlah baris dan kolom).  
#  - Menampilkan 5 baris pertama untuk gambaran awal struktur dan isi data.
# 

# %%

# ======================
# 2. LOAD DATA
# ======================
df = pd.read_csv('../data/heart.csv')
print(f"Shape dataset: {df.shape}")
print("\n5 data pertama:")
display(df.head())


# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)
# Tahap ini bertujuan untuk memahami karakteristik dataset melalui analisis dan visualisasi.  
# - Mengecek tipe data, statistik deskriptif, missing values, distribusi, dan korelasi.  
# - Tujuannya: Mengidentifikasi pola, anomali, dan hubungan antar fitur.
# 

# %% [markdown]
# ### 3.1. Info Dasar
# - `df.info()`: Menampilkan tipe data dan jumlah non-null per kolom untuk memahami struktur dataset.  
# - `df.describe()`: Menyediakan statistik deskriptif (mean, min, max, dll.) untuk fitur numerik.
# 

# %%
# ======================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ======================
print("\n=== EKSPLORASI DATA ===")

# 3.1. Info Dasar
print("\nInfo dataset:")
print(df.info())

print("\nStatistik deskriptif:")
df.describe()

#copy describe



# %% [markdown]
# 
# ### 3.2. Cek Missing Values
# Memeriksa jumlah nilai hilang (missing values) per kolom untuk mengevaluasi kualitas data.  
# - Hasil menunjukkan apakah ada data yang perlu diimputasi atau ditangani.
# 

# %%

# 3.2. Cek Missing Values
print("\nMissing values per kolom:")
print(df.isnull().sum())



# %% [markdown]
# ### 3.3. Analisis Target
# Visualisasi distribusi kolom target `HeartDisease` menggunakan countplot.  
# - Tujuannya: Memahami proporsi kelas (0 = sehat, 1 = sakit) untuk mengecek keseimbangan data.
# 

# %%

# 3.3. Analisis Target
plt.figure(figsize=(6,4))
sns.countplot(x='HeartDisease', data=df)
plt.title('Distribusi Target (HeartDisease)')
plt.show()


# %% [markdown]
# 
# ### 3.4. Analisis Fitur Numerik
# Menampilkan distribusi fitur numerik (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) dengan histogram dan KDE.  
# - Tujuannya: Memahami pola distribusi dan mendeteksi potensi anomali seperti nilai ekstrim.
# 

# %%
num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
print("\nAnalisis fitur numerik:")

for col in num_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Liat rata-rata
print("\nRata-rata fitur numerik:")
for col in num_features:
    mean_value = df[col].mean()
    print(f"Rata-rata {col}: {mean_value:.2f}")

# mayoritas fitur numerik
print("\nMayoritas fitur numerik:")

for col in num_features:
    mode_value = df[col].mode()[0]
    print(f"Mayoritas {col}: {mode_value:.2f}")


# %% [markdown]
# ### 3.5. Analisis Outlier
# Menggunakan boxplot untuk mendeteksi outlier pada fitur numerik.  
# - Boxplot menunjukkan nilai ekstrim yang mungkin memengaruhi performa model jika tidak ditangani.
# 

# %%

# Boxplot untuk outlier
plt.figure(figsize=(15,10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()


# %% [markdown]
# 
# ### 3.6. Analisis Fitur Kategorik
# Visualisasi distribusi fitur kategorik (`Sex`, `ChestPainType`, dll.) terhadap target `HeartDisease` menggunakan countplot.  
# - Tujuannya: Mengidentifikasi hubungan antara kategori dan risiko penyakit jantung.
# 

# %%

cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

print("\nAnalisis fitur kategorik:")

for col in cat_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='HeartDisease', data=df)
    plt.title(f'Distribusi {col} vs HeartDisease')
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='HeartDisease')
    plt.show()


# %%

# berapa persen categori yang menderita penyakit jantung vs tidak masing masing categori
print("\nPersentase HeartDisease per kategori:")
for col in cat_features:
    percentages = df.groupby(col)['HeartDisease'].value_counts(normalize=True).unstack() * 100
    print(f"\nPersentase HeartDisease per kategori {col}:")
    print(percentages)

# %% [markdown]
# ### 3.7. Korelasi Antar Fitur
# Heatmap korelasi menunjukkan hubungan antar fitur numerik dan target.  
# - Warna merah (positif) dan biru (negatif) mengindikasikan kekuatan korelasi.  
# - Tujuannya: Mengidentifikasi fitur yang paling berpengaruh terhadap `HeartDisease`.
# 

# %%

# 3.6. Korelasi Antar Fitur
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi Antar Fitur Numerik")
plt.show()


# %% [markdown]
# ### 3.8. Cek Duplicate Value
# 

# %%
df.duplicated()

# %% [markdown]
# 
# 
# ### 📌 **Insight Ringkas Dataset Kesehatan Jantung (918 data)**
# 
# 1. **Usia Pasien:**
#    Rata-rata usia adalah **53,5 tahun**, mayoritas antara **47–60 tahun**, menunjukkan fokus pada kelompok usia paruh baya.
# 
# 2. **Tekanan Darah & Kolesterol:**
# 
#    * Rata-rata **RestingBP**: 132 mmHg → agak tinggi.
#    * Rata-rata **Cholesterol**: 198 mg/dL, tapi ada nilai **0** → kemungkinan data kosong/salah.
# 
# 3. **Detak Jantung Maksimum (MaxHR):**
#    Rata-rata **137 bpm**, dengan minimum **60 bpm** dan maksimum **202 bpm** → variasi besar.
# 
# 4. **Penderita Penyakit Jantung:**
#    Kolom `HeartDisease` menunjukkan **\~55% pasien terdiagnosis** (mean = 0.55) → dataset agak seimbang untuk klasifikasi.
# 
# 5. **Data Kategorikal:**
#    Ada 5 kolom bertipe objek (seperti `Sex`, `ChestPainType`) → penting untuk encoding saat modeling.
# 
# 6. **Outlier & Anomali:**
# 
#    * Nilai **0 pada RestingBP & Cholesterol** tidak logis → perlu dibersihkan.
#    * Nilai **Oldpeak negatif (-2.6)** juga tidak wajar.
# 
# 
# 

# %% [markdown]
# ## 4. Data Preprocessing
# Tahap ini mempersiapkan data untuk pemodelan dengan menangani nilai tidak valid, missing values, outlier, dan encoding.  
# - Tujuannya: Memastikan data bersih, konsisten, dan siap untuk pelatihan model.
# 

# %% [markdown]
# ### 4.1. Handle Invalid Values
# Menangani nilai tidak valid berdasarkan domain medis:  
# - `Cholesterol` = 0 dan `RestingBP` = 0 dianggap tidak mungkin, diganti dengan NaN.  
# - `Oldpeak` negatif tidak wajar, di-clip ke minimum 0.

# %%

# ======================
# 4. DATA PREPROCESSING
# ======================
print("\n=== PREPROCESSING ===")


# 4.1. Handle Invalid Values
print("\n1. Menangani Nilai Tidak Valid:")

# Cholesterol = 0 tidak mungkin secara medis -> anggap missing value
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

# Oldpeak negatif tidak valid -> clamp ke minimum 0
df['Oldpeak'] = df['Oldpeak'].clip(lower=0)

# RestingBP = 0 tidak mungkin -> anggap missing value
df['RestingBP'] = df['RestingBP'].replace(0, np.nan)

print("✅ Nilai tidak valid pada Cholesterol, Oldpeak, dan RestingBP sudah ditangani")


# %% [markdown]
# ### 4.2. Handle Missing Values
# Mengisi nilai hilang (NaN) pada fitur numerik dengan strategi median.  
# - Median dipilih karena lebih tahan terhadap outlier dibandingkan mean.  
# - Tujuannya: Memastikan tidak ada data hilang untuk pemodelan.
# 

# %%

# 4.2. Handle Missing Values
print("\n2. Imputasi Missing Values:")

# Imputasi dengan median 
imputer = SimpleImputer(strategy='median')
df[num_features] = imputer.fit_transform(df[num_features])

print("✅ Missing values diimputasi dengan median")


# %% [markdown]
# ### 4.3. Handle Outliers
# Menangani outlier pada fitur numerik dengan clipping berdasarkan rentang realistis (domain medis).  
# - Contoh: `Age` di-clip ke [30, 100], `Cholesterol` ke [100, 400].  
# - Tujuannya: Mengurangi dampak nilai ekstrim tanpa menghapus data.
# 

# %%

# 4.3. Handle Outliers (Disesuaikan dengan Domain Medis)
print("\n3. Penanganan Outlier yang Realistis:")

outlier_ranges = {
    'Age': (30, 100),          # Usia pasien realistis
    'RestingBP': (80, 200),    # Tekanan darah normal/tidak
    'Cholesterol': (100, 400), # Range kolesterol medis
    'MaxHR': (60, 220),        # Detak jantung manusia
    'Oldpeak': (0, 4)          # Depresi ST tidak negatif
}

for col in num_features:
    if col in outlier_ranges:
        min_val, max_val = outlier_ranges[col]
        df[col] = np.clip(df[col], min_val, max_val)
        print(f" - {col}: di-clip ke range [{min_val}, {max_val}]")


# %% [markdown]
# ### 4.4. Encoding Kategorikal
# Mengubah fitur kategorikal menjadi numerik menggunakan one-hot encoding.  
# - `drop_first=True` untuk menghindari multicollinearity.  
# - Tujuannya: Membuat data kompatibel dengan model machine learning.
# 

# %%

# 4.4. Encoding Kategorikal (Sama seperti Awal)
print("\n4. Encoding Variabel Kategorikal (Original):")


df = pd.get_dummies(df, columns=cat_features, drop_first=True)

print("✅ One-hot encoding untuk semua fitur kategorikal")


# %% [markdown]
# ### 4.5. Split Data
# Membagi data menjadi fitur (X) dan target (y), lalu ke data latih (80%) dan uji (20%).  
# - `stratify=y` memastikan proporsi kelas target seimbang di kedua set.  
# - Tujuannya: Menyediakan data untuk pelatihan dan evaluasi model.
# 

# %%

# 4.4. Split Data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %% [markdown]
# ### 4.6. Scaling
# Menstandarisasi fitur numerik menggunakan StandardScaler.  
# - Mengubah data ke skala mean=0 dan std=1.  
# - Tujuannya: Memastikan fitur numerik memiliki skala yang sama untuk model.
# 

# %%

# 4.5. Scaling
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# %% [markdown]
# ## 5. Modeling
# Tahap ini membangun model klasifikasi untuk memprediksi `HeartDisease`.  
# - Model: Logistic Regression, dipilih karena sederhana dan interpretatif.
# 

# %%

# ======================
# 5. MODELING
# ======================
print("\n=== PEMBANGUNAN MODEL ===")


# %% [markdown]
# ### 5.1. Inisialisasi Model
# Menginisialisasi model Logistic Regression.  
# - `max_iter=1000`: Memastikan konvergensi.  
# - `class_weight='balanced'`: Menangani ketidakseimbangan kelas.  
# - `random_state=42`: Konsistensi hasil.
# 

# %%

# 5.1. Inisialisasi Model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')


# %% [markdown]
# ### 5.2. Training
# Melatih model menggunakan data latih (X_train, y_train).  
# - Tujuannya: Model mempelajari hubungan antara fitur dan target.

# %%

# 5.2. Training
model.fit(X_train, y_train)


# %% [markdown]
# ### 5.3. Prediksi
# Melakukan prediksi pada data uji.  
# - `y_pred`: Prediksi kelas (0 atau 1).  
# - `y_proba`: Probabilitas untuk kelas positif (1).
# 

# %%

# 5.3. Prediksi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# %% [markdown]
# ## 6. Evaluasi Model
# Mengevaluasi performa model dengan metrik dan visualisasi.  
# - Tujuannya: Menilai akurasi, presisi, dan kemampuan model secara keseluruhan.
# 

# %%

# ======================
# 6. EVALUASI MODEL
# ======================
print("\n=== EVALUASI MODEL ===")


# %% [markdown]
# ### 6.1. Accuracy
# Menghitung akurasi, yaitu proporsi prediksi benar dari total prediksi.  
# - Hasil menunjukkan seberapa baik model secara umum.
# 

# %%

# 6.1. Accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")


# %% [markdown]
# ### 6.2. Classification Report
# Menampilkan metrik presisi, recall, dan F1-score per kelas.  
# - Tujuannya: Memahami performa model untuk setiap kelas (0 dan 1).
# 

# %%

# 6.2. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %% [markdown]
# ### 6.3. Confusion Matrix
# Visualisasi matriks kebingungan untuk melihat prediksi benar dan salah.  
# - TP, TN, FP, FN membantu menganalisis kesalahan model.

# %%

# 6.3. Confusion Matrix
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# %% [markdown]
# ### 6.4. ROC Curve
# Menampilkan kurva ROC dan skor AUC.  
# - ROC menunjukkan trade-off antara TPR dan FPR.  
# - AUC mendekati 1 berarti performa model baik.

# %%

# 6.4. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# %% [markdown]
# ### 6.5. Feature Importance
# Menampilkan 10 fitur teratas berdasarkan koefisien absolut model.  
# - Tujuannya: Memahami fitur yang paling berpengaruh terhadap prediksi.
# 

# %%

# 6.5. Feature Importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=coefficients.head(10))
plt.title('Top 10 Feature Importance')
plt.show()


# %% [markdown]
# ## 7. Prediksi Data Baru
# Menguji model pada data baru untuk simulasi penggunaan nyata.  
# - Data baru diproses (encoding, scaling) sebelum prediksi.  
# - Menampilkan hasil prediksi dan probabilitas.

# %% [markdown]
# 
# Fungsi untuk memproses data baru agar sesuai format model.  
# - Melakukan encoding, menambahkan kolom hilang, dan scaling.
# 

# %% [markdown]
# 
# Menguji model pada data contoh pasien.  
# - Menampilkan hasil prediksi (kelas) dan probabilitas risiko penyakit jantung.
# 

# %%

# ======================
# 7. PREDIKSI DATA BARU
# ======================
print("\n=== CONTOH PREDIKSI ===")

# Fungsi untuk mempersiapkan data baru
def prepare_new_data(input_data):
    # Convert input to DataFrame
    new_df = pd.DataFrame([input_data])
    
    # Handle categorical variables
    new_df = pd.get_dummies(new_df)
    
    # Ensure all columns exist
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = 0
    
    # Reorder columns
    new_df = new_df[X.columns]
    
    # Scale numerical features
    new_df[num_features] = scaler.transform(new_df[num_features])
    
    return new_df

# Contoh data input
sample_data = {
    'Age': 58,
    'RestingBP': 140,
    'Cholesterol': 289,
    'FastingBS': 0,
    'MaxHR': 160,
    'Oldpeak': 1.2,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingECG': 'Normal',
    'ExerciseAngina': 'N',
    'ST_Slope': 'Up'
}

# Preprocess data baru
new_data = prepare_new_data(sample_data)

# Prediksi
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]

print("\nHasil Prediksi:")
print(f"Input Data: {sample_data}")
print(f"Prediksi: {'Penyakit Jantung' if prediction[0] == 1 else 'Tidak Ada Penyakit Jantung'}")
print(f"Probabilitas: {probability[0]*100:.2f}%")


# %% [markdown]
# ## 8. Simpan Model
# Menyimpan model dan objek preprocessing untuk penggunaan di masa depan.  
# - Disimpan dalam format `.pkl` di folder `models`.  
# - Tujuannya: Memudahkan deployment atau prediksi ulang.

# %%

# ======================
# 8. SIMPAN MODEL
# ======================
import joblib
import datetime

# Simpan model dan preprocessing objects
model_data = {
    'model': model,
    'scaler': scaler,
    'imputer': imputer,
    'features': X.columns.tolist(),
    'num_features': num_features,
    'date': datetime.datetime.now().strftime("%Y-%m-%d")
}

# Simpan model ke folder models buat jika belum ada
if not os.path.exists('../models'):
    os.makedirs('../models')
joblib.dump(model_data, '../models/heart_disease_model.pkl')
print("\nModel berhasil disimpan!")



