# 🚕 Robotaxi Data Analytics & Executive Dashboard

Proyek analisis data komprehensif (*End-to-End*) untuk layanan taksi otonom (Robotaxi). Proyek ini mencakup tahapan *Business Understanding*, *Data Cleaning*, *Exploratory Data Analysis* (EDA), hingga divisualisasikan dalam bentuk *Interactive Dashboard* bergaya profesional.

## 📂 Struktur Folder
*   `robotaxi_rides.csv`: Dataset operasional dan transaksional perjalanan.
*   `robotaxi_users.csv`: Dataset profil demografi dan loyalitas pengguna.
*   `robotaxi.ipynb`: Jupyter Notebook yang berisi langkah-langkah pembersihan data dan eksplorasi data (EDA).
*   `eda_robotaxi.py`: Skrip Python mandiri untuk menampilkan ringkasan statistik.
*   `dashboard.py`: Skrip utama untuk menjalankan *Executive Dashboard* interaktif menggunakan Plotly Dash.
*   `requirements.txt`: Daftar pustaka (library) Python yang dibutuhkan.

## 💡 Key Business Insights (Temuan Utama)
Berdasarkan hasil analisis, berikut adalah 3 temuan bisnis yang krusial:
1. **Operasional:** Waktu tunggu rata-rata (~6 menit) terbukti **bukan** penyebab utama pembatalan pesanan. Pembatalan (*Cancellation Rate* 12.1%) lebih mungkin disebabkan oleh faktor eksternal penumpang.
2. **Segmentasi Marketing:** Terdapat anomali pada program loyalitas; pengguna **Bronze** (Tier terendah) justru menghasilkan *Average Revenue Per User* (ARPU) dan Persentase Tip **tertinggi** dibandingkan tier Diamond. Strategi *reward* perusahaan perlu dievaluasi.
3. **Strategi Harga (Surge Pricing):** Menerapkan lonjakan harga tinggi (>2.0x) **tidak merusak** tingkat kesuksesan perjalanan, dan anehnya, justru mendapatkan *rating* pelanggan tertinggi (4.34). Algoritma harga saat ini terbukti sangat optimal di jam sibuk.

## 🚀 Cara Menjalankan Dashboard
1. Pastikan Python sudah terpasang. Install semua *library* yang dibutuhkan dengan perintah:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan skrip *dashboard*:
   ```bash
   python dashboard.py
   ```
3. Buka browser web Anda (Chrome/Edge/Safari) dan kunjungi: **`http://127.0.0.1:8050`**

---
*Proyek ini merupakan simulasi penyelesaian masalah bisnis dunia nyata (Business Intelligence) untuk mengoptimalkan operasional dan pendapatan.*
