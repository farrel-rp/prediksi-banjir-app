# 🌊 Sistem Analisis Prediktif Potensi Banjir

Proyek ini adalah sistem *machine learning* berbasis **Random Forest** yang dirancang untuk memprediksi apakah suatu area berpotensi banjir berdasarkan sejumlah parameter cuaca dan lingkungan. Sistem ini terdiri dari dua komponen utama: **skrip pelatihan model** (`model_training.py`) dan **aplikasi web interaktif** menggunakan Streamlit (`app.py`).

---

## 📁 Struktur Proyek

```
production/
│
├── dataset/
│   ├── data_dummy_banjir_train.csv   # 50.000 baris data pelatihan
│   └── data_dummy_banjir_test.csv    # 100 baris data pengujian
│
├── model_training.py                 # Skrip pelatihan model
├── app.py                            # Aplikasi prediksi (Streamlit)
├── requirements.txt                  # Dependensi Python
├── random_forest_model.pkl           # Model terlatih (dihasilkan otomatis)
└── encoder_cuaca.pkl                 # LabelEncoder cuaca (dihasilkan otomatis)
```

---

## 🗂️ Dataset

Dataset yang digunakan adalah **data dummy** yang dibangkitkan secara sintetis menggunakan `generate.py`. Dataset dirancang agar realistis secara logika: kondisi cuaca mempengaruhi curah hujan, yang kemudian berdampak pada ketinggian sungai dan debit air.

### Fitur (Kolom Input)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Curah Hujan (mm)` | Numerik | Intensitas curah hujan per jam |
| `Ketinggian Sungai (cm)` | Numerik | Ketinggian permukaan sungai |
| `Debit Air (m³/s)` | Numerik | Volume air yang mengalir per detik |
| `Luas Resapan (%)` | Numerik | Persentase area yang dapat menyerap air |
| `Drainase (skor 1-10)` | Numerik | Kualitas sistem drainase (1=Buruk, 10=Sangat Baik) |
| `Kondisi Cuaca` | Kategorikal | `Hujan Ringan` / `Hujan Sedang` / `Hujan Berat` |

### Target (Label)

| Nilai | Representasi Biner | Makna |
|---|---|---|
| `"Banjir"` | `1` | Area berpotensi banjir |
| `"Tidak"` | `0` | Area tidak berpotensi banjir |

### Logika Pembangkitan Label

Label pada dataset ditentukan menggunakan rumus skor bahaya vs. skor ketahanan:

```
score_bahaya    = (Curah Hujan × 0.4) + (Ketinggian Sungai × 0.3) + (Debit Air × 0.1)
score_ketahanan = (Luas Resapan × 0.5) + (Drainase × 5.0)

Jika score_bahaya > (score_ketahanan + 40) → Label = "Banjir"
Jika sebaliknya                             → Label = "Tidak"
```

Logika ini mencerminkan bahwa:
- **Hujan lebat, sungai meluap, dan debit tinggi** → meningkatkan risiko.
- **Lahan resapan luas dan drainase baik** → meningkatkan ketahanan area.

---

## 🔄 Alur Pelatihan Model (`model_training.py`)

Berikut adalah alur lengkap proses pelatihan model dari awal hingga penyimpanan:

```
┌─────────────────────┐
│  1. LOAD DATA       │  Membaca train.csv (50.000 baris) & test.csv (100 baris)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. PREPROCESSING   │  a. LabelEncoder: "Hujan Ringan/Sedang/Berat" → angka
└────────┬────────────┘  b. Label Mapping: "Banjir"→1, "Tidak"→0
         │
         ▼
┌─────────────────────┐
│  3. MODEL TRAINING  │  RandomForestClassifier.fit(X_train, y_train)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. EVALUASI        │  Accuracy, Classification Report, Confusion Matrix
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. SERIALISASI     │  joblib.dump → random_forest_model.pkl + encoder_cuaca.pkl
└─────────────────────┘
```

### Langkah 1 — Load Data

```python
base_dir   = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'dataset/data_dummy_banjir_train.csv')
test_path  = os.path.join(base_dir, 'dataset/data_dummy_banjir_test.csv')
```

Path dibuat **relatif terhadap lokasi skrip** (`__file__`), bukan terhadap direktori kerja terminal. Hal ini memastikan skrip dapat dijalankan dari direktori manapun tanpa error `FileNotFoundError`.

### Langkah 2 — Preprocessing

**a. Encoding Fitur Kategorikal (`Kondisi Cuaca`)**

`LabelEncoder` dari scikit-learn mengubah string menjadi angka integer agar model dapat memprosesnya:

```
"Hujan Berat"  → 0
"Hujan Ringan" → 1
"Hujan Sedang" → 2
```

Penting: encoder hanya di-*fit* pada data **training**, lalu di-*transform* pada data testing:

```python
X_train['Kondisi Cuaca'] = encoder_cuaca.fit_transform(X_train['Kondisi Cuaca'])
X_test['Kondisi Cuaca']  = encoder_cuaca.transform(X_test['Kondisi Cuaca'])  # ← bukan fit_transform!
```

> Menggunakan `fit_transform` pada data testing akan menyebabkan **data leakage** — model seolah-olah "mengintip" distribusi data testing saat training, yang membuat evaluasi tidak valid.

**b. Konversi Target Label ke Biner**

```python
label_mapping = {'Tidak': 0, 'Banjir': 1}
y_train = y_train_raw.map(label_mapping)
y_test  = y_test_raw.map(label_mapping)
```

### Langkah 3 — Pelatihan Model

Model yang dipilih adalah **Random Forest Classifier**, sebuah ensemble dari banyak *Decision Tree* yang bekerja secara paralel dan hasilnya digabungkan dengan *majority voting*.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

Lihat penjelasan lengkap setiap parameter di bawah.

### Langkah 4 — Evaluasi

Model dievaluasi menggunakan tiga metrik utama:

| Metrik | Penjelasan |
|---|---|
| **Accuracy** | Persentase prediksi yang benar secara keseluruhan |
| **Classification Report** | Precision, Recall, F1-Score per kelas |
| **Confusion Matrix** | Jumlah True Positive, True Negative, False Positive, False Negative |

**Hasil evaluasi terakhir:**

```
Accuracy: 0.9827

Classification Report:
              precision    recall  f1-score   support

       Tidak       0.98      0.98      0.98      5216
      Banjir       0.98      0.98      0.98      4784

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000


Confusion Matrix:
[[5137   79]
 [  94 4690]]
```

**Interpretasi Confusion Matrix:**

```
                Prediksi: Tidak   Prediksi: Banjir
Aktual: Tidak      5137 (TN)          79 (FP)
Aktual: Banjir       94 (FN)        4690 (TP)
```

- **94 kasus False Negative** (area banjir diprediksi aman) — ini lebih kritis secara praktis.
- **79 kasus False Positive** (area aman diprediksi banjir) — cukup rendah, performa model sangat baik secara keseluruhan.

### Langkah 5 — Serialisasi

Model dan encoder disimpan menggunakan `joblib` agar dapat dimuat kembali oleh `app.py` tanpa perlu melatih ulang:

```python
joblib.dump(rf_model,       'random_forest_model.pkl')
joblib.dump(encoder_cuaca,  'encoder_cuaca.pkl')
```

---

## ⚙️ Penjelasan Teknis Parameter Random Forest

Pemilihan nilai parameter pada `RandomForestClassifier` bukan sembarangan. Setiap nilai dipilih berdasarkan pertimbangan teknis untuk **menyeimbangkan antara akurasi dan generalisasi** (mencegah overfitting).

### `n_estimators=100` — Jumlah Pohon

Random Forest bekerja dengan membangun banyak *Decision Tree* secara independen pada subset data yang berbeda (proses *bootstrap sampling*), lalu menggabungkan prediksi mereka (*majority voting*).

**Mengapa 100?**
- Terlalu sedikit (misal: 10) → variansi prediksi tinggi, hasil tidak stabil.
- Terlalu banyak (misal: 1000) → waktu komputasi jauh lebih lama tanpa peningkatan akurasi yang signifikan.
- **100 adalah titik keseimbangan standar industri** yang cukup untuk dataset ≤ 100.000 baris.

### `max_depth=10` — Batas Kedalaman Pohon

Kedalaman pohon menentukan seberapa kompleks setiap *Decision Tree* yang dibentuk. Pohon yang terlalu dalam akan **menghafal (overfitting)** data training, termasuk noise-nya.

**Mengapa 10?**
- Dengan 6 fitur input, kedalaman 10 sudah lebih dari cukup untuk menangkap pola yang kompleks.
- Membatasi kedalaman memaksa setiap pohon untuk membuat aturan yang lebih **general**, bukan menghafalkan data secara detail.
- Tanpa batasan (`max_depth=None`), pohon bisa tumbuh sangat dalam hingga setiap daun hanya berisi 1 sampel — overfitting sempurna.

### `min_samples_split=5` — Sampel Minimum untuk Split

Parameter ini mengatur berapa minimal jumlah sampel di sebuah node agar node tersebut bisa dipecah menjadi dua cabang.

**Mengapa 5?**
- Nilai default adalah 2 (setiap node dengan ≥2 sampel bisa di-split).
- Dengan nilai 5, node yang hanya berisi ≤4 sampel tidak akan dipecah lagi → pohon tidak membuat aturan yang terlalu spesifik hanya karena segelintir data.
- Pada dataset 50.000 baris, 5 sampel adalah ambang batas yang **sangat konservatif namun efektif** untuk mencegah split yang tidak bermakna.

### `min_samples_leaf=2` — Sampel Minimum di Daun

Berbeda dengan `min_samples_split`, parameter ini mengatur berapa minimal sampel yang **harus ada di daun (leaf)** setelah sebuah split terjadi.

**Mengapa 2?**
- Nilai 1 (default) berarti sebuah daun boleh berisi hanya 1 sampel → pohon bisa sangat terspesialisasi.
- Nilai 2 memastikan setiap keputusan akhir pohon didukung oleh **minimal 2 data poin**, bukan hanya anomali tunggal.
- Dikombinasikan dengan `min_samples_split=5`, ini menciptakan efek "pruning ringan" yang sangat efektif.

### `random_state=42` — Seed Acak

Random Forest melibatkan banyak proses stokastik: pemilihan subset data (*bagging*) dan pemilihan subset fitur di setiap split.

**Mengapa 42?**
- Tanpa seed, setiap kali skrip dijalankan akan menghasilkan model yang **sedikit berbeda** karena proses acak yang berbeda.
- Dengan `random_state=42`, hasil pelatihan menjadi **100% reproducible** — tim lain yang menjalankan skrip yang sama akan mendapatkan model dan evaluasi yang identik.
- Nilai `42` adalah konvensi populer di komunitas data science, tidak ada keistimewaan matematis, namun sudah menjadi standar de facto.

---

## 🚀 Cara Menjalankan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Latih Ulang Model

Jalankan skrip pelatihan untuk menghasilkan file `.pkl` terbaru:

```bash
python model_training.py
```

### 3. Jalankan Aplikasi Web

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Komponen | Library |
|---|---|
| Data Processing | `pandas` |
| Machine Learning | `scikit-learn` |
| Model Serialization | `joblib` |
| Web Interface | `streamlit` |

---

## 📊 Mengapa Random Forest?

Random Forest dipilih karena beberapa keunggulan yang relevan dengan kasus ini:

1. **Robust terhadap Outlier** — Ensemble banyak pohon mengurangi dampak satu data anomali.
2. **Tidak memerlukan feature scaling** — Fitur numerik seperti curah hujan dan debit air tidak perlu dinormalisasi.
3. **Menangani fitur kategorikal** (setelah encoding) dengan baik.
4. **Feature importance** — Dapat dianalisis untuk memahami fitur mana yang paling berpengaruh terhadap prediksi banjir.
5. **Performa tinggi** pada dataset tabular ukuran sedang (10.000–500.000 baris) tanpa tuning yang rumit.
