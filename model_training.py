import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

# ---------------------------------------------------------
# 1. Environment Setup & Data Loading
# ---------------------------------------------------------
print("Membaca data training dan testing...")
# Memuat dataset (path diasumsikan berada di direktori parent)
# Mendapatkan path absolut dari direktori tempat skrip ini berada
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset/data_dummy_banjir_train.csv')

# Membaca full dataset 50,000
df = pd.read_csv(dataset_path)

# Memisahkan fitur (X) dan target (y)
X = df.drop('Label', axis=1)
y_raw = df['Label']

# Membagi dataset 80% training, 20% testing
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
print("Melakukan preprocessing data...")

# Menggunakan LabelEncoder untuk fitur 'Kondisi Cuaca'
encoder_cuaca = LabelEncoder()

# Untuk mencegah SettingWithCopyWarning dari pandas
X_train = X_train.copy()
X_test = X_test.copy()

# Fit encoder pada data training dan transform, kemudian transform pada data testing saja
# agar tidak terjadi data leakage (kebocoran informasi dari data testing)
X_train['Kondisi Cuaca'] = encoder_cuaca.fit_transform(X_train['Kondisi Cuaca'])
X_test['Kondisi Cuaca'] = encoder_cuaca.transform(X_test['Kondisi Cuaca'])

# Mengubah target Label menjadi biner (0 dan 1)
# Asumsi: "Tidak" = 0, "Banjir" = 1
label_mapping = {'Tidak': 0, 'Banjir': 1}
y_train = y_train_raw.map(label_mapping)
y_test = y_test_raw.map(label_mapping)

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
print("Melatih model Random Forest...")

# Inisialisasi model Random Forest Classifier
# Menambahkan parameter dasar untuk mencegah overfitting:
# - max_depth: membatasi kedalaman maksimum pohon
# - min_samples_split: jumlah sampel minimum untuk split node
# - min_samples_leaf: jumlah sampel minimum pada setiap leaf (daun)
rf_model = RandomForestClassifier(
    n_estimators=100,        # Jumlah pohon yang akan dibuat
    max_depth=10,            # Kedalaman maksimal pohon (mencegah pohon terlalu spesifik terhadap data training)
    min_samples_split=5,     # Minimal ada 5 sampel untuk cabang bisa dibagi lagi
    min_samples_leaf=2,      # Minimal ada 2 sampel di tiap ujung cabang (daun)
    random_state=42          # Seed agar hasil pelatihan konsisten (reproducible)
)

# Melatih (training) model menggunakan data training
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Model Evaluation
# ---------------------------------------------------------
print("Mengevaluasi model pada data testing...")

# Melakukan prediksi pada data testing
y_pred = rf_model.predict(X_test)

# Menampilkan hasil evaluasi model (Akurasi, Laporan Klasifikasi, dan Confusion Matrix)
print("\n--- Hasil Evaluasi Model ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
# target_names memperjelas label mana yang merepresentasikan 0 dan 1
print(classification_report(y_test, y_pred, target_names=['Tidak', 'Banjir']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 5. Model Serialization (Export)
# ---------------------------------------------------------
print("\nMenyimpan model dan encoder...")

# Menentukan lokasi folder production untuk menyimpan model
production_dir = os.path.join(base_dir, '../production')
os.makedirs(production_dir, exist_ok=True)

# Menyimpan model Random Forest ke dalam file .pkl di folder production
joblib.dump(rf_model, os.path.join(production_dir, 'random_forest_model.pkl'))

# Menyimpan encoder_cuaca ke dalam file .pkl di folder production
joblib.dump(encoder_cuaca, os.path.join(production_dir, 'encoder_cuaca.pkl'))

print("Proses selesai!")
print("Model Random Forest telah disimpan sebagai 'random_forest_model.pkl'.")
print("Encoder untuk Kondisi Cuaca telah disimpan sebagai 'encoder_cuaca.pkl'.")
