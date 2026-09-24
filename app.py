import streamlit as st
import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from deepface import DeepFace
import database as db

# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Sistem Presensi SMK PGRI Sukoharjo",
    page_icon="🏫",
    layout="wide"
)

db.init_db()

# ============================================================
# FUNGSI BANTUAN (ALJABAR, PREPROCESSING & HELPER)
# ============================================================
def hitung_jarak_cosine(vektor1, vektor2):
    """Menghitung jarak Cosine: 1 - Cosine Similarity"""
    v1 = np.array(vektor1, dtype=np.float64)
    v2 = np.array(vektor2, dtype=np.float64)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 1.0
    return float(1.0 - (dot / norm))

def hitung_jarak_euclidean(vektor1, vektor2):
    """Menghitung jarak Euclidean"""
    v1 = np.array(vektor1, dtype=np.float64)
    v2 = np.array(vektor2, dtype=np.float64)
    return float(np.linalg.norm(v1 - v2))

def prapemrosesan_citra(uploaded_file):
    """Prapemrosesan: Mempertahankan rasio aspek agar wajah tidak terdistorsi"""
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)
    temp_path = os.path.abspath("temp_proses.jpg")
    img.save(temp_path, format="JPEG", quality=95)
    return temp_path, img

def ambil_vektor_secara_aman(siswa_record, model_target):
    """Fungsi penyelamat: Mengambil vektor secara aman (kompatibel format lama & baru)"""
    # Jika format data baru multi-model (vektor_dict)
    if "vektor_dict" in siswa_record and isinstance(siswa_record["vektor_dict"], dict):
        return siswa_record["vektor_dict"].get(model_target)
    
    # Jika format data lama (hanya ada key 'vektor')
    if "vektor" in siswa_record:
        return siswa_record["vektor"]
        
    return None

# ============================================================
# BILAH SISI (SIDEBAR) & PENGATURAN MODEL
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/school.png", width=70)
st.sidebar.title("SMK PGRI Sukoharjo")
st.sidebar.markdown("**Sistem Presensi Biometrik Wajah**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi Menu:",
    ["📷 Presensi Siswa", "📋 Log Daftar Kehadiran", "➕ Registrasi Siswa Baru", "👥 Data Siswa Terdaftar"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan Evaluasi Model")

model_pilihan = st.sidebar.selectbox(
    "Arsitektur Model Aktif:",
    ["ArcFace (Rekomendasi)", "VGG-Face", "FaceNet"]
)

if "ArcFace" in model_pilihan:
    model_name = "ArcFace"
    metric_type = "cosine"
    default_thresh = 0.5000  # Disetel 0.5000 agar lebih ramah variasi kacamata
    max_thresh = 1.0000
elif "VGG-Face" in model_pilihan:
    model_name = "VGG-Face"
    metric_type = "cosine"
    default_thresh = 0.6800
    max_thresh = 1.0000
else:
    model_name = "Facenet"
    metric_type = "euclidean"
    default_thresh = 10.0000
    max_thresh = 20.0000

threshold = st.sidebar.slider(
    f"Ambang Batas ({metric_type.capitalize()}):",
    min_value=0.0000,
    max_value=max_thresh,
    value=default_thresh,
    step=0.0100
)

st.sidebar.caption(f"Model Aktif: **{model_name}** | Metrik: **{metric_type.capitalize()}**")

# ============================================================
# MODUL 1: PRESENSI SISWA
# ============================================================
if menu == "📷 Presensi Siswa":
    st.title("📷 Pindai Presensi Wajah Siswa")
    st.write("Silakan masukkan citra wajah siswa untuk melakukan pencatatan kehadiran otomatis.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Masukan Citra")
        metode_input = st.radio("Metode Pengambilan:", ["Gunakan Kamera Laptop", "Unggah File Foto"], horizontal=True)
        
        uploaded_image = None
        if metode_input == "Gunakan Kamera Laptop":
            uploaded_image = st.camera_input("Ambil foto siswa untuk presensi:")
        else:
            uploaded_image = st.file_uploader("Pilih foto siswa (.JPG, .PNG):", type=["jpg", "jpeg", "png"])
            
    with col2:
        st.subheader("Hasil Identifikasi")
        if uploaded_image is not None:
            temp_path, img_display = prapemrosesan_citra(uploaded_image)
            st.image(img_display, caption="Citra Masukan Presensi", width=280)
            
            with st.spinner(f"Mengekstrak fitur menggunakan {model_name}..."):
                try:
                    start_time = time.time()
                    embedding_objs = DeepFace.represent(
                        img_path=temp_path,
                        model_name=model_name,
                        detector_backend="mtcnn",
                        enforce_detection=False
                    )
                    vektor_uji = embedding_objs[0]["embedding"]
                    durasi_ekstraksi = (time.time() - start_time) * 1000
                    
                    daftar_siswa = db.ambil_semua_siswa()
                    
                    if len(daftar_siswa) == 0:
                        st.warning("Basis data masih kosong! Daftarkan siswa terlebih dahulu di menu 'Registrasi Siswa Baru'.")
                    else:
                        jarak_terkecil = float("inf")
                        siswa_cocok = None
                        ada_vektor_valid = False
                        
                        for s in daftar_siswa:
                            # Mengambil vektor secara aman (bebas KeyError)
                            vektor_ref = ambil_vektor_secara_aman(s, model_name)
                            
                            if vektor_ref is None:
                                continue
                                
                            # Cek kesesuaian dimensi
                            if len(vektor_uji) != len(vektor_ref):
                                continue
                                
                            ada_vektor_valid = True
                            if metric_type == "cosine":
                                jarak = hitung_jarak_cosine(vektor_uji, vektor_ref)
                            else:
                                jarak = hitung_jarak_euclidean(vektor_uji, vektor_ref)
                                
                            if jarak < jarak_terkecil:
                                jarak_terkecil = jarak
                                siswa_cocok = s
                        
                        st.markdown("---")
                        if not ada_vektor_valid:
                            st.warning(f"Data siswa di database belum memiliki representasi untuk model {model_name}. Silakan daftarkan ulang siswa di menu Registrasi agar ketiga model terisi otomatis.")
                        elif siswa_cocok is not None and jarak_terkecil <= threshold:
                            st.success(f"### IDENTITAS DIKENALI: {siswa_cocok['nama_siswa']}")
                            st.info(f"**NISN:** {siswa_cocok['nisn']} | **Kelas:** {siswa_cocok['kelas']}")
                            st.write(f"Model: **{model_name}** | Skor Jarak: `{jarak_terkecil:.4f}` (Ambang Batas: `{threshold:.4f}`)")
                            st.write(f"Waktu Pemrosesan: `{durasi_ekstraksi:.2f} ms`")
                            
                            sukses, pesan = db.catat_presensi(
                                id_siswa=siswa_cocok["id_siswa"],
                                nama_siswa=siswa_cocok["nama_siswa"],
                                model=model_name,
                                skor_jarak=jarak_terkecil
                            )
                            if sukses:
                                st.balloons()
                                st.success(pesan)
                            else:
                                st.info(pesan)
                        else:
                            st.error("### AKSES DITOLAK: WAJAH TIDAK DIKENALI")
                            st.write(f"Skor Jarak Terdekat: `{jarak_terkecil:.4f}` (Melampaui Ambang Batas `{threshold:.4f}`)")
                            st.caption("Tips: Pastikan pencahayaan cukup dan wajah menghadap lurus ke kamera.")
                            
                except Exception as e:
                    st.error(f"Gagal memproses pengenalan wajah: {e}")
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

# ============================================================
# MODUL 2: LOG DAFTAR KEHADIRAN (TABEL PRESENSI)
# ============================================================
elif menu == "📋 Log Daftar Kehadiran":
    st.title("📋 Log Riwayat Presensi Siswa")
    st.write("Rekapitulasi pencatatan kehadiran siswa yang tersimpan di dalam basis data SQLite.")
    
    df_presensi = db.ambil_log_presensi_df()
    
    if len(df_presensi) > 0:
        st.dataframe(df_presensi, use_container_width=True)
        
        csv_data = df_presensi.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Data Presensi (Format CSV)",
            data=csv_data,
            file_name=f"Presensi_SMK_PGRI_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada data presensi yang tercatat hari ini.")

# ============================================================
# MODUL 3: REGISTRASI SISWA BARU (AUTO MULTI-MODEL)
# ============================================================
elif menu == "➕ Registrasi Siswa Baru":
    st.title("➕ Pendaftaran Identitas Siswa Baru")
    st.info("Sistem akan secara otomatis mengekstrak vektor untuk 3 model sekaligus (ArcFace, VGG-Face, dan FaceNet) dari satu foto ini.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Diri Siswa")
        nisn = st.text_input("Nomor Induk Siswa Nasional (NISN):", placeholder="Contoh: 0071234567")
        nama = st.text_input("Nama Lengkap Siswa:", placeholder="Contoh: Budi Santoso")
        kelas = st.selectbox("Tingkat Kelas:", ["X RPL", "XI RPL", "XII RPL", "X TKJ", "XI TKJ", "XII TKJ"])
        
        st.markdown("---")
        st.subheader("Foto Referensi Siswa")
        metode_daftar = st.radio("Metode Pengambilan Foto:", ["Gunakan Kamera Laptop (Webcam)", "Unggah File Foto"], horizontal=True)
        
        foto_daftar = None
        if metode_daftar == "Gunakan Kamera Laptop (Webcam)":
            foto_daftar = st.camera_input("Ambil foto normal siswa untuk database:")
        else:
            foto_daftar = st.file_uploader("Pilih foto wajah normal siswa (.JPG, .PNG):", type=["jpg", "jpeg", "png"])

        tombol_daftar = st.button("Daftarkan Siswa ke Basis Data", type="primary")

    with col2:
        st.subheader("Status Pendaftaran")
        if tombol_daftar:
            if not nisn or not nama or foto_daftar is None:
                st.warning("Mohon lengkapi data NISN, Nama, dan ambil/unggah foto wajah siswa terlebih dahulu!")
            else:
                temp_path, img_display = prapemrosesan_citra(foto_daftar)
                st.image(img_display, caption=f"Foto Pendaftaran: {nama}", width=260)
                
                with st.spinner("Mengekstrak vektor fitur untuk ArcFace, VGG-Face, dan FaceNet secara serentak..."):
                    try:
                        vektor_dict = {}
                        daftar_arsitektur = ["ArcFace", "VGG-Face", "Facenet"]
                        
                        for m in daftar_arsitektur:
                            res_emb = DeepFace.represent(
                                img_path=temp_path,
                                model_name=m,
                                detector_backend="mtcnn",
                                enforce_detection=False
                            )
                            vektor_dict[m] = res_emb[0]["embedding"]
                        
                        sukses, pesan = db.simpan_siswa_multi_model(nisn, nama, kelas, vektor_dict)
                        if sukses:
                            st.success(pesan)
                            st.info("Profil berhasil disimpan untuk 3 model sekaligus! Anda bebas berganti model di sidebar tanpa perlu daftar ulang.")
                        else:
                            st.error(pesan)
                    except Exception as e:
                        st.error(f"Gagal mengekstrak fitur wajah: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass

# ============================================================
# MODUL 4: DATA SISWA TERDAFTAR (DILENGKAPI FITUR HAPUS)
# ============================================================
elif menu == "👥 Data Siswa Terdaftar":
    st.title("👥 Data Siswa Terdaftar")
    daftar_siswa = db.ambil_semua_siswa()
    
    if len(daftar_siswa) > 0:
        data_tabel = []
        pilihan_siswa = {}
        
        for s in daftar_siswa:
            if "vektor_dict" in s and isinstance(s["vektor_dict"], dict):
                model_info = ", ".join(list(s["vektor_dict"].keys()))
            else:
                model_info = "Format Tunggal (ArcFace)"
                
            data_tabel.append({
                "ID": s["id_siswa"],
                "NISN": s["nisn"],
                "Nama Siswa": s["nama_siswa"],
                "Kelas": s["kelas"],
                "Model Tersedia": model_info
            })
            
            # Format label dropdown: Nama Siswa - NISN
            label_siswa = f"{s['nama_siswa']} (NISN: {s['nisn']}) - ID: {s['id_siswa']}"
            pilihan_siswa[label_siswa] = s["id_siswa"]
            
        st.dataframe(pd.DataFrame(data_tabel), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🗑️ Pengelolaan Data: Hapus Siswa")
        st.write("Pilih siswa yang ingin dihapus dari basis data referensi:")
        
        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            siswa_terpilih = st.selectbox("Pilih Siswa:", list(pilihan_siswa.keys()))
        
        with col_del2:
            st.write("") # Spacer agar tombol sejajar dengan selectbox
            st.write("")
            tombol_hapus = st.button("Hapus Data Siswa", type="primary")
            
        if tombol_hapus:
            id_yang_dihapus = pilihan_siswa[siswa_terpilih]
            db.hapus_siswa(id_yang_dihapus)
            st.success(f"Data '{siswa_terpilih}' dan seluruh riwayat presensinya berhasil dihapus!")
            time.sleep(1)
            st.rerun() # Muat ulang tampilan tabel secara otomatis
            
    else:
        st.info("Belum ada siswa yang terdaftar di basis data.")