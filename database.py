import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime

DB_NAME = "presensi_smk.db"

def get_connection():
    """Membuka koneksi ke SQLite"""
    return sqlite3.connect(DB_NAME)

def init_db():
    """Inisialisasi tabel basis data relasional"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Tabel Referensi Siswa (Menyimpan Vektor Identitas Hasil Transfer Learning)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS siswa_referensi (
            id_siswa INTEGER PRIMARY KEY AUTOINCREMENT,
            nisn TEXT UNIQUE NOT NULL,
            nama_siswa TEXT NOT NULL,
            kelas TEXT NOT NULL,
            vektor_wajah TEXT NOT NULL,
            tanggal_daftar TEXT NOT NULL
        )
    ''')
    
    # 2. Tabel Log Kehadiran (Mencatat Riwayat Presensi)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS log_kehadiran (
            id_presensi INTEGER PRIMARY KEY AUTOINCREMENT,
            id_siswa INTEGER,
            nama_siswa TEXT NOT NULL,
            tanggal TEXT NOT NULL,
            waktu_masuk TEXT NOT NULL,
            status TEXT NOT NULL,
            model_digunakan TEXT NOT NULL,
            skor_jarak REAL,
            FOREIGN KEY (id_siswa) REFERENCES siswa_referensi (id_siswa)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database SQLite 'presensi_smk.db' dan tabel relasional berhasil disiapkan.")

def simpan_siswa(nisn, nama_siswa, kelas, vektor_embedding):
    """Menyimpan data siswa dan vektor embedding (disimpan sebagai string JSON)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    vektor_str = json.dumps(vektor_embedding.tolist() if isinstance(vektor_embedding, np.ndarray) else vektor_embedding)
    tanggal_daftar = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        cursor.execute('''
            INSERT INTO siswa_referensi (nisn, nama_siswa, kelas, vektor_wajah, tanggal_daftar)
            VALUES (?, ?, ?, ?, ?)
        ''', (nisn, nama_siswa, kelas, vektor_str, tanggal_daftar))
        conn.commit()
        sukses = True
        pesan = f"Siswa {nama_siswa} (NISN: {nisn}) berhasil didaftarkan!"
    except sqlite3.IntegrityError:
        sukses = False
        pesan = f"Error: NISN {nisn} sudah terdaftar di database."
    finally:
        conn.close()
        
    return sukses, pesan

def ambil_semua_siswa():
    """Mengambil semua siswa terdaftar beserta vektor wajahnya untuk pencocokan"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id_siswa, nisn, nama_siswa, kelas, vektor_wajah FROM siswa_referensi')
    rows = cursor.fetchall()
    conn.close()
    
    daftar_siswa = []
    for r in rows:
        daftar_siswa.append({
            "id_siswa": r[0],
            "nisn": r[1],
            "nama_siswa": r[2],
            "kelas": r[3],
            "vektor": np.array(json.loads(r[4]))
        })
    return daftar_siswa

def catat_presensi(id_siswa, nama_siswa, model="ArcFace", skor_jarak=0.0):
    """Mencatat kehadiran jika siswa belum presensi pada hari yang sama"""
    conn = get_connection()
    cursor = conn.cursor()
    
    sekarang = datetime.now()
    tanggal_hari_ini = sekarang.strftime("%Y-%m-%d")
    waktu_sekarang = sekarang.strftime("%H:%M:%S")
    
    # Cek apakah siswa sudah presensi hari ini
    cursor.execute('''
        SELECT id_presensi FROM log_kehadiran 
        WHERE id_siswa = ? AND tanggal = ?
    ''', (id_siswa, tanggal_hari_ini))
    
    sudah_absen = cursor.fetchone()
    
    if sudah_absen:
        conn.close()
        return False, f"{nama_siswa} sudah melakukan presensi hari ini!"
    
    # Jika belum, catat kehadiran
    cursor.execute('''
        INSERT INTO log_kehadiran (id_siswa, nama_siswa, tanggal, waktu_masuk, status, model_digunakan, skor_jarak)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (id_siswa, nama_siswa, tanggal_hari_ini, waktu_sekarang, "Hadir", model, skor_jarak))
    
    conn.commit()
    conn.close()
    return True, f"Presensi berhasil dicatat: {nama_siswa} ({waktu_sekarang})"

def ambil_log_presensi_df():
    """Mengambil seluruh data log presensi dalam bentuk DataFrame Pandas"""
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT id_presensi as "ID", nama_siswa as "Nama Siswa", tanggal as "Tanggal", 
               waktu_masuk as "Jam Masuk", status as "Status", 
               model_digunakan as "Model", ROUND(skor_jarak, 4) as "Skor Jarak (Cosine)"
        FROM log_kehadiran ORDER BY id_presensi DESC
    ''', conn)
    conn.close()
    return df

def hapus_siswa(id_siswa):
    """Menghapus data siswa dan seluruh riwayat presensinya berdasarkan ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Hapus riwayat presensi terkait di log_kehadiran terlebih dahulu
    cursor.execute('DELETE FROM log_kehadiran WHERE id_siswa = ?', (id_siswa,))
    
    # 2. Hapus data profil dan vektor di siswa_referensi
    cursor.execute('DELETE FROM siswa_referensi WHERE id_siswa = ?', (id_siswa,))
    
    conn.commit()
    conn.close()
    return True

if __name__ == "__main__":
    init_db()