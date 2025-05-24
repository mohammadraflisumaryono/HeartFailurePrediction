
# Prediksi Penyakit Jantung dengan Logistic Regression

Proyek ini bertujuan untuk membangun model machine learning berbasis **Logistic Regression** untuk memprediksi risiko penyakit jantung berdasarkan data medis pasien.

## 📖 Deskripsi
Model dikembangkan menggunakan dataset berisi informasi kesehatan pasien, dengan fitur seperti usia, tekanan darah istirahat, kolesterol, detak jantung maksimal, dan faktor risiko lainnya. Model akan memprediksi apakah seseorang berisiko mengalami penyakit jantung.

## 🏗️ Alur Proyek
1. **Eksplorasi Data**: Memeriksa distribusi data, missing value, dan outlier.
2. **Preprocessing**:  
   - Penanganan missing value  
   - Encoding variabel kategorikal  
   - Standarisasi fitur numerik  
   - Split data menjadi training dan testing
3. **Modeling**:  
   - Inisialisasi model Logistic Regression dengan `class_weight='balanced'`  
   - Pelatihan model menggunakan data training
4. **Evaluasi Model**:  
   - Menghitung **Accuracy**  
   - Menampilkan **Classification Report** (Precision, Recall, F1)  
   - Menampilkan **Confusion Matrix**  
   - Menampilkan **ROC Curve dan AUC Score**  
   - Analisis **Feature Importance**
5. **Inference (Prediksi Data Baru)**:  
   - Menyiapkan data baru dan melakukan preprocessing  
   - Memprediksi kelas dan probabilitas  
   - Menampilkan hasil prediksi dan confidence

## 📊 Visualisasi
- Confusion Matrix (heatmap)  
- ROC Curve dengan AUC  
- Grafik Feature Importance (Top 10 fitur)

## 🛠️ Cara Menjalankan
1. Clone repositori ini:
    ```bash
    git clone https://github.com/username/heart-disease-prediction.git
    cd heart-disease-prediction
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Jalankan notebook analisis (gunakan Jupyter Notebook atau VSCode):
    ```bash
    jupyter notebook HeartDisease_Prediction.ipynb
    ```
4. Lihat hasil prediksi, evaluasi, dan inference.

## 📝 Contoh Hasil Prediksi Data Baru
**Input Data Baru**:
| Age | RestingBP | Cholesterol | FastingBS | MaxHR | Oldpeak | Sex | ChestPainType | RestingECG | ExerciseAngina | ST_Slope |
|-----|-----------|-------------|-----------|-------|---------|-----|---------------|-------------|----------------|----------|
| 58  | 140       | 289         | 0         | 160   | 1.2     | M   | ATA           | Normal      | N              | Up       |

**Hasil Prediksi**: Penyakit Jantung  
**Probabilitas Positif**: 85.73%  
**Probabilitas Negatif**: 14.27%

## 📂 Struktur Proyek
````

heart-disease-prediction/
├── data/
│   └── heart\_disease\_data.csv
├── models/
│   └── logistic\_regression\_model.pkl
├── notebooks/
│   └── HeartDisease\_Prediction.ipynb
├── requirements.txt
└── README.md

```

## 📌 Dependencies Utama
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn


## 📢 Catatan
- Model ini adalah baseline menggunakan Logistic Regression. Bisa ditingkatkan dengan model lain (Random Forest, XGBoost, dll) dan hyperparameter tuning.
- Data diolah secara bertanggung jawab untuk menjaga akurasi dan fairness.

---
