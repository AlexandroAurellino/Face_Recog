# 🏫 Sistem Presensi Wajah Siswa SMK PGRI Berbasis Transfer Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-DeepFace-orange.svg)](https://github.com/serengil/deepface)
[![Database](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)

Purwarupa (_prototype_) aplikasi web presensi biometrik pengenalan wajah nirkontak yang dikembangkan sebagai bagian dari penelitian Tugas Akhir Skripsi di **Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Kristen Duta Wacana (UKDW)**.

Aplikasi ini mengimplementasikan metode **Transfer Learning** (ekstraksi fitur murni dengan bobot beku) untuk membandingkan kinerja tiga arsitektur model _pre-trained_ ternama (**ArcFace**, **VGG-Face**, dan **FaceNet**) terhadap variasi visual siswa (seperti penggunaan hijab, kacamata, masker, dan pencahayaan kelas).

---

## 📌 Fitur Utama

- **Pindai Presensi Wajah (Webcam & Unggah Foto):** Memproses deteksi dan verifikasi identitas siswa secara instan (< 1 detik).
- **Multi-Model Support & Evaluasi Sidang:** Menyediakan opsi perbandingan model langsung di bilah sisi (_sidebar_) antara ArcFace, VGG-Face, dan FaceNet.
- **Registrasi Sekali untuk Semua Model:** Pendaftaran siswa baru otomatis mengekstrak representasi vektor untuk ketiga arsitektur secara serentak ke dalam basis data.
- **Slider Ambang Batas Dinamis (_Threshold Tuning_):** Memungkinkan penguji mengkalibrasi nilai ambang batas secara interaktif untuk menguji variasi atribut wajah.
- **Pencatatan Kehadiran Otomatis:** Menyimpan rekaman kehadiran (_timestamp_, tanggal, nama siswa, dan skor jarak) ke basis data SQLite nirserver.
- **Ekspor Laporan CSV:** Mengunduh rekapitulasi data kehadiran siswa ke format `.csv` dengan satu klik.
- **Manajemen Data Siswa:** Menampilkan daftar siswa terdaftar dan menyediakan fitur penghapusan data secara aman.

---

## 🛠️ Arsitektur & Teknologi

| Komponen                | Teknologi / Arsitektur                     | Keterangan                                                   |
| :---------------------- | :----------------------------------------- | :----------------------------------------------------------- |
| **Bahasa Pemrograman**  | Python 3.10+                               | Bahasa utama pengembangan sistem                             |
| **Model Evaluasi (AI)** | ArcFace, VGG-Face, FaceNet                 | Model _pre-trained_ pengekstraksi fitur wajah (_embeddings_) |
| **Detektor Wajah**      | MTCNN (_Multi-Task Cascaded CNN_)          | Lokalisasi wajah dan estimasi 5 titik tengara (_landmarks_)  |
| **Metrik Jarak**        | _Cosine Similarity_ & _Euclidean Distance_ | Perhitungan probabilitas kecocokan vektor wajah              |
| **Antarmuka Pengguna**  | Streamlit                                  | Kerangka kerja aplikasi web interaktif                       |
| **Basis Data**          | SQLite3                                    | Penyimpanan relasional lokal untuk vektor dan log presensi   |

---

## 📂 Struktur Direktori Proyek

```text
├── .gitignore               # Berkas pengabaian Git (mengabaikan .venv dan cache)
├── README.md                # Dokumentasi proyek
├── requirements.txt         # Daftar dependensi Python
├── database.py              # Modul pengelola basis data SQLite
├── app.py                   # Berkas utama aplikasi antarmuka Streamlit
└── presensi_smk.db          # Berkas basis data SQLite (dihasilkan otomatis)
```
