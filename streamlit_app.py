import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile
import io
import gc
import urllib.request

# Konfigurasi Halaman Web
st.set_page_config(page_title="Image Classification via ZIP", layout="centered")

# Nama file model sesuai script asli Anda
MODEL_FILE = "load_image_classification_model.h5"

# =========================================================================
# PASTE LINK GOOGLE DRIVE KAMU DI BAWAH INI (Ganti teks di dalam tanda kutip)
# =========================================================================
GDrive_Link = "https://drive.google.com/file/d/1U5coUaAXjAWRXbxtuMGzkLpz8NUeUkgQ/view?usp=sharing"

# Fungsi untuk mengubah link share Google Drive biasa menjadi link download langsung
def get_direct_download_link(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://docs.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_resource
def load_model_from_drive():
    # Jika file model belum ada di server lokal Streamlit, download dari Google Drive
    if not os.path.exists(MODEL_FILE):
        if GDrive_Link == "MASUKKAN_LINK_GOOGLE_DRIVE_KAMU_DI_SINI" or GDrive_Link == "":
            st.warning("⚠️ Kamu belum memasukkan link Google Drive di dalam kode script!")
            return None
        
        with st.spinner("⏳ Mengunduh file model h5 dari Google Drive (Hanya dilakukan sekali saat pertama kali dibuka)..."):
            try:
                direct_link = get_direct_download_link(GDrive_Link)
                urllib.request.urlretrieve(direct_link, MODEL_FILE)
                st.success("✅ Model berhasil diunduh dari Google Drive!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model dari Google Drive: {e}")
                return None
                
    try:
        return tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"Gagal memuat file model h5: {e}")
        return None

# Memuat model otomatis dari Google Drive
model = load_model_from_drive()

st.title("🖼️ Image Classification via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto beton untuk melakukan klasifikasi secara otomatis.")

# Spesifikasi target size dan label persis sesuai file asli Anda
target_size = (128, 128)
class_labels = {0: "negative", 1: "positive"}

if model is not None:
    st.success("✅ Model AI berhasil dimuat dan siap digunakan!")

# Widget pengunggahan khusus berkas ZIP
uploaded_zip = st.file_uploader(
    "Pilih dan unggah file ZIP berisi kumpulan foto...", 
    type=["zip"]
)

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.info("📦 Berkas ZIP terdeteksi! Mengekstrak isi file...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Saring file gambar asli dan eliminasi file sampah sistem tersembunyi
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) 
                and not f.startswith('__MACOSX/') 
                and not os.path.basename(f).startswith('.')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("⚠️ Tidak ditemukan file gambar (.jpg/.png) yang valid di dalam ZIP Anda.")
            else:
                st.success(f"🚀 Menemukan {total_images} gambar. Memulai klasifikasi otomatis...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                # Proses iterasi setiap gambar di dalam ZIP
                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    
                    try:
                        img_data = z.read(file_name)
                        with Image.open(io.BytesIO(img_data)) as img:
                            # Preprocessing persis mengikuti formula file asal Anda
                            image_rgb = img.convert("RGB")
                            img_resized = image_rgb.resize(target_size)
                            img_array = np.array(img_resized, dtype=np.float32) / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                        
                        # Eksekusi Prediksi Model
                        prediction = model.predict(img_array, verbose=0)
                        probability = float(prediction[0][0])

                        # Logika Klasifikasi Ambang Batas 0.5 sesuai file asli Anda
                        if probability > 0.5:
                            predicted_class = 1
                            confidence = probability * 100
                        else:
                            predicted_class = 0
                            confidence = (1 - probability) * 100
                        
                        label = class_labels[predicted_class]
                        
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Prediksi": label.upper(),
                            "Tingkat Keyakinan": f"{confidence:.2f}%",
                            "Raw Probability": f"{probability:.4f}"
                        })
                        
                    except Exception:
                        continue
                    
                    # Update progress bar
                    progress_bar.progress((idx + 1) / total_images)
                    
                    # Manajemen RAM server agar tidak memicu Out-Of-Memory (OOM)
                    if (idx + 1) % 5 == 0:
                        tf.keras.backend.clear_session()
                        gc.collect()

                status_text.empty()
                
                # Menampilkan representasi data tabel hasil akhir
                if results:
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    total_pos = sum(1 for r in results if r["Prediksi"] == "POSITIVE")
                    total_neg = sum(1 for r in results if r["Prediksi"] == "NEGATIVE")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Kategori Positive (Retak)", total_pos)
                    col2.metric("Total Kategori Negative (Aman)", total_neg)
                        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file ZIP: {e}")
