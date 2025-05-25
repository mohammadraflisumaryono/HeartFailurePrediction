# %%
# %% [markdown]
# # Prediksi Penyakit Jantung (Heart Disease Prediction)
# **Dataset**: Heart Failure Prediction (918 sampel)  
# **Jenis**: Klasifikasi biner  
# **Target**: HeartDisease (0 = sehat, 1 = sakit)


# %%
# ======================
# 1. IMPORT LIBRARY
# ======================
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


# %%

# ======================
# 2. LOAD DATA
# ======================
df = pd.read_csv('../data/heart.csv')
print(f"Shape dataset: {df.shape}")
print("\n5 data pertama:")
display(df.head())


# %%

# ======================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ======================
print("\n=== EKSPLORASI DATA ===")


# %%

# 3.1. Info Dasar
print("\nInfo dataset:")
print(df.info())

print("\nStatistik deskriptif:")
df.describe()

#copy describe



# %%

# 3.2. Cek Missing Values
print("\nMissing values per kolom:")
print(df.isnull().sum())



# %%

# 3.3. Analisis Target
plt.figure(figsize=(6,4))
sns.countplot(x='HeartDisease', data=df)
plt.title('Distribusi Target (HeartDisease)')
plt.show()


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

# %%

# Boxplot untuk outlier
plt.figure(figsize=(15,10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()


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

# 3.6. Korelasi Antar Fitur
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi Antar Fitur Numerik")
plt.show()


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


# %%

# 4.2. Handle Missing Values
print("\n2. Imputasi Missing Values:")

# Imputasi dengan median (tanpa membuat fitur baru)
imputer = SimpleImputer(strategy='median')
df[num_features] = imputer.fit_transform(df[num_features])

print("✅ Missing values diimputasi dengan median")


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


# %%

# 4.4. Encoding Kategorikal (Sama seperti Awal)
print("\n4. Encoding Variabel Kategorikal (Original):")


df = pd.get_dummies(df, columns=cat_features, drop_first=True)

print("✅ One-hot encoding untuk semua fitur kategorikal")


# %%

# 4.4. Split Data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%

# 4.5. Scaling
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# %%

# ======================
# 5. MODELING
# ======================
print("\n=== PEMBANGUNAN MODEL ===")


# %%

# 5.1. Inisialisasi Model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')


# %%

# 5.2. Training
model.fit(X_train, y_train)


# %%

# 5.3. Prediksi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# %%

# ======================
# 6. EVALUASI MODEL
# ======================
print("\n=== EVALUASI MODEL ===")


# %%

# 6.1. Accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")


# %%

# 6.2. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%

# 6.3. Confusion Matrix
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


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



