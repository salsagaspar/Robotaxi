# 🚕 Robotaxi Data Analytics & Executive Dashboard

Proyek analisis data komprehensif (*End-to-End*) untuk layanan taksi otonom (Robotaxi). Proyek ini mencakup tahapan *Business Understanding*, *Data Cleaning*, *Exploratory Data Analysis* (EDA), integrasi model *Machine Learning*, hingga divisualisasikan dalam bentuk *Interactive Dashboard* bergaya profesional.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python 3
* **Data Analisis & Manipulasi:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Visualisasi & Dashboard:** Plotly, Dash, Dash Bootstrap Components
* **Eksplorasi Data:** Jupyter Notebook

## 📂 Struktur Folder
*   `robotaxi_rides.csv`: Dataset operasional dan transaksional perjalanan.
*   `robotaxi_users.csv`: Dataset profil demografi dan loyalitas pengguna.
*   `robotaxi.ipynb`: Jupyter Notebook yang berisi langkah-langkah pembersihan data dan eksplorasi data (EDA).
*   `train_models.py`: Skrip untuk melatih 5 model Machine Learning (Random Forest) dan menyimpannya.
*   `models/`: Direktori tempat penyimpanan model ML yang sudah dilatih (`.pkl`).
*   `dashboard.py`: Skrip utama untuk menjalankan *Executive Dashboard* interaktif dan *AI Control Center* menggunakan Plotly Dash.
*   `requirements.txt`: Daftar pustaka (library) Python yang dibutuhkan.

## 💡 Key Business Insights (Temuan Utama)
Berdasarkan hasil analisis, berikut adalah 3 temuan bisnis yang krusial:
1. **Operasional:** Waktu tunggu rata-rata (~6 menit) terbukti **bukan** penyebab utama pembatalan pesanan. Pembatalan (*Cancellation Rate* 12.1%) lebih mungkin disebabkan oleh faktor eksternal penumpang.
2. **Segmentasi Marketing:** Terdapat anomali pada program loyalitas; pengguna **Bronze** (Tier terendah) justru menghasilkan *Average Revenue Per User* (ARPU) dan Persentase Tip **tertinggi** dibandingkan tier Diamond. Strategi *reward* perusahaan perlu dievaluasi.
3. **Strategi Harga (Surge Pricing):** Menerapkan lonjakan harga tinggi (>2.0x) **tidak merusak** tingkat kesuksesan perjalanan, dan anehnya, justru mendapatkan *rating* pelanggan tertinggi (4.34). Algoritma harga saat ini terbukti sangat optimal di jam sibuk.

## 🤖 Machine Learning Integrations (AI Control Center)
Proyek ini sekarang dilengkapi dengan 5 model *Machine Learning* berbasis **Random Forest** untuk beralih dari analisis deskriptif menjadi prediktif:
1. **Demand Forecasting:** Memprediksi jumlah permintaan taksi untuk 24 jam ke depan di berbagai kota (Time Series).
2. **Dynamic Pricing Predictor:** Menghitung prediksi estimasi harga perjalanan berdasarkan jarak, durasi, dan waktu penjemputan.
3. **Predictive Maintenance:** Mengklasifikasi apakah sebuah mobil Robotaxi membutuhkan perawatan berdasarkan *battery level*, *mileage*, dan sensor tekanan ban.
4. **Customer Churn Risk:** Memprediksi probabilitas seorang pengguna akan berhenti menggunakan layanan berdasarkan usia, total perjalanan, dan saldo promo.
5. **ETA Predictor:** Memprediksi Estimasi Waktu Kedatangan (durasi perjalanan) berdasarkan jarak tempuh dan jam sibuk.

Semua fitur prediktif ini dapat diakses secara interaktif melalui tab **AI Models** di dalam Dashboard Plotly Dash.

## 🚀 Cara Menjalankan Proyek
1. Pastikan Python sudah terpasang. Install semua *library* yang dibutuhkan dengan perintah:
   ```bash
   pip install -r requirements.txt
   ```
2. **(Opsional)** Jika Anda ingin melatih ulang model *Machine Learning* atau folder `models/` belum ada, jalankan:
   ```bash
   python train_models.py
   ```
3. Jalankan skrip *dashboard*:
   ```bash
   python dashboard.py
   ```
4. Buka browser web Anda (Chrome/Edge/Safari) dan kunjungi: **`http://127.0.0.1:8050`**

---
*Proyek ini merupakan simulasi penyelesaian masalah bisnis dunia nyata (Business Intelligence) untuk mengoptimalkan operasional dan pendapatan.*
