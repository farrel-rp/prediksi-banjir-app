import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. Memuat Model dan Encoder
# ---------------------------------------------------------
# Pastikan file .pkl berada di direktori yang sama dengan app.py
rf_model = joblib.load('random_forest_model.pkl')
encoder_cuaca = joblib.load('encoder_cuaca.pkl')

# ---------------------------------------------------------
# 2. Pengaturan Halaman dan Judul UI
# ---------------------------------------------------------
st.set_page_config(page_title="Prediksi Potensi Banjir", layout="centered")
st.title("Sistem Analisis Prediktif Potensi Banjir 🌊")
st.write("Masukkan parameter lingkungan di bawah ini untuk memprediksi apakah suatu area berpotensi banjir.")

# ---------------------------------------------------------
# 3. Form Input Pengguna
# ---------------------------------------------------------
st.header("Parameter Input")

col1, col2 = st.columns(2)

with col1:
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0, max_value=500, value=50)
    ketinggian_sungai = st.number_input("Ketinggian Sungai (cm)", min_value=0, max_value=500, value=50)
    debit_air = st.number_input("Debit Air (m³/s)", min_value=0, max_value=1000, value=100)

with col2:
    luas_resapan = st.number_input("Luas Resapan (%)", min_value=0, max_value=100, value=50)
    drainase = st.slider("Skor Drainase (1 Buruk - 10 Sangat Baik)", min_value=1, max_value=10, value=5)
    # Gunakan list label asli dari encoder_cuaca agar pilihan dinamis dan akurat
    opsi_cuaca = encoder_cuaca.classes_
    kondisi_cuaca = st.selectbox("Kondisi Cuaca", opsi_cuaca)

# ---------------------------------------------------------
# 4. Tombol Prediksi dan Pemrosesan
# ---------------------------------------------------------
if st.button("Prediksi Potensi Banjir", type="primary"):
    
    # Encode kondisi cuaca yang dipilih pengguna ke dalam format angka
    cuaca_encoded = encoder_cuaca.transform([kondisi_cuaca])[0]
    
    # Susun input pengguna menjadi DataFrame agar sesuai dengan format saat training
    input_data = pd.DataFrame({
        'Curah Hujan (mm)': [curah_hujan],
        'Ketinggian Sungai (cm)': [ketinggian_sungai],
        'Debit Air (m³/s)': [debit_air],
        'Luas Resapan (%)': [luas_resapan],
        'Drainase (skor 1-10)': [drainase],
        'Kondisi Cuaca': [cuaca_encoded]
    })
    
    # Lakukan prediksi menggunakan model Random Forest
    prediksi = rf_model.predict(input_data)[0]
    
    # ---------------------------------------------------------
    # 5. Tampilkan Hasil
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("Hasil Prediksi")
    
    if prediksi == 1:
        st.error("⚠️ **WASPADA: Area ini BERPOTENSI BANJIR dengan kondisi saat ini.**")
    else:
        st.success("✅ **AMAN: Area ini TIDAK BERPOTENSI BANJIR dengan kondisi saat ini.**")
