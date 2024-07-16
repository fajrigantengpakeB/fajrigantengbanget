import streamlit as st  # type: ignore # Library untuk membuat antarmuka web
import cv2  # type: ignore # Library untuk pengolahan gambar
import numpy as np  # type: ignore # Library untuk komputasi numerik
import time  # Library untuk mengelola waktu
from keras.models import load_model  # type: ignore # Library untuk memuat model Keras
from keras.preprocessing import image  # type: ignore # Library untuk pemrosesan gambar
from PIL import Image, ImageOps  # type: ignore # Library untuk manipulasi gambar

# Menetapkan ikon dan judul halaman
icon_path = "labels.txt"
st.set_page_config(page_title="predict_kidney_stone", page_icon=icon_path)

# Memuat model yang telah dilatih
model = load_model("keras_model.h5")

def load_and_process_image(img_path, target_size=(224,224, 3)):
    """Memuat dan memproses gambar"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array /= 255.0  # Normalisasi
    return img_array
def colored_divider(color, thickness="15px"):
    st.markdown(f"<hr style='border: none; border-top: {thickness} solid {color};' />", unsafe_allow_html=True)

def large_tittle(text, size="55px"): 
    st.markdown(f"<h1 style='font-size: {size};'>{text}</h1>", unsafe_allow_html=True)
def large_text(text, size="20px"):
    st.markdown(f"<p style='font-size: {size};'>{text}</p>", unsafe_allow_html=True)

def predict_image_class(model, img_path, class_labels):
    """Melakukan prediksi kelas gambar menggunakan model"""
    processed_image = load_and_process_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    skor_kepercayaan = float(prediction[0][predicted_class])
    return predicted_label, skor_kepercayaan

# Daftar label kelas (pastikan urutannya sesuai dengan class_indices pada training)
class_labels = ['Normal','Batu Ginjal']

# Aplikasi Streamlit
# Navigasi

halaman_terpilih = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Penjelasan", "Gejala", "Penyebab","Deteksi"], format_func=lambda x: x)

if halaman_terpilih == "Beranda":
    # Tampilkan halaman beranda
    large_tittle("WELCOME APLIKASI DETEKSI BATU GINJAL 🫀")
    colored_divider(color="red")    # Menambahkan garis pemisah berwarna
    st.write(
        "Aplikasi ini memungkinkan anda untuk mendeteksi adanya penyakit batu ginjal atau tidak adanya penyakit batu ginjal (ginjal normal) berdasarkan hasil gambar CT-Scan ginjal "
    )
    st.write(
        "Untuk mulai pemeriksaan silahkan ke menu (deteksi)."
    )
elif halaman_terpilih == "Penjelasan": 
     large_tittle("PENJELASAN BATU GINJAL")
     colored_divider(color="red")
     st.image("C:\\Users\\ASUS\\Fotoginjal.jpg")
     large_text("Batu ginjal adalah batu yang terbentuk di tubuli ginjal kemudian berada di kaliks, infundibulum, pelvis ginjal dan bahkan bisa mengisi pelvis serta seluruh kaliks ginjal dan merupakan batu saluran kemih yang paling sering terjadi. Penyebab terbentuknya batu saluran kemih diduga berhubungan dengan gangguan aliran urine, gangguan metabolik, infeksi saluran kemih.")

elif halaman_terpilih == "Gejala": 
     large_tittle("TANDA DAN GEJALA BATU GINJAL")
     colored_divider(color="red")
     large_text("Gejala yang muncul pada penyakit batu ginjal bervariasi tergantung ukuran pembentukan batu pada ginjal. Gejala umum yang muncul di antaranya:")
     large_text("Adanya nyeri pada punggung atau nyeri kolik yang hebat, Nyeri kolik ditandai dengan rasa sakit yang hilang timbul di sekitar tulang rusuk dan pinggang kemudian menjalar ke bagian perut dan daerah paha sebelah dalam, Adanya nyeri hebat biasa diikuti demam dan menggigil, Kemungkinan adanya rasa mual dan terjadinya muntah dan gangguan perut, Adanya darah di dalam urin dan adanya gangguan buang air kecil, penderita juga sering BAK atau malah terjadinya penyumbatan pada saluran kemih, Jika ini terjadi maka resiko terjadinya infeksi saluran kemih menjadi lebih besar.") # type: ignore # type: ignore

elif halaman_terpilih == "Penyebab": 
     large_tittle("PENYEBAB BATU GINJAL")
     colored_divider(color="red")
     large_text("Banyak faktor yang bisa menyebabkan batu. Dalam sistem urinaria (sekresi), ginjal berfungsi untuk melakukan penyaringan pada darah. Penyaringan ini berfungsi untuk menyeimbangkan kadar mineral tubuh. Di saat tubuh kekurangan air, ginjal menyumplai air melalui darah. Selebihnya dibuang melalui ureter ke kandung kemih dalam bentuk urin. Begitupun dengan jenis mineral lain, vitamin-vitamin, kalsium dan zat-zat lainnya. Vitamin memang dibutuhkan tubuh tetapi jika terlalu banyak vitamin harus dikeluarkan oleh ginjal.")

elif halaman_terpilih == "Deteksi":
    # Tampilkan halaman deteksi
    st.title("Unggah Gambar")
    colored_divider(color="red")
    st.markdown("---")

    # Unggah gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silakan pilih gambar", type=["jpg", "jpeg", "png"])
    
    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang diunggah", use_column_width=True)
        if st.button("Deteksi"):
            with st.spinner('Medeteksi gambar...'):
                # Simpan berkas gambar yang diunggah ke lokasi sementara
                with open("temp_image.jpg", "wb") as f:
                    f.write(berkas_gambar.getbuffer())

                # Lakukan prediksi pada berkas yang disimpan
            predicted_label, skor_kepercayaan = predict_image_class(model, "temp_image.jpg", class_labels)

            # Tampilkan hasil prediksi
            st.write(f"Hasil Deteksi : {predicted_label}")
            st.write(f"Skor Kepercayaan: {skor_kepercayaan * 100:.2f}%")

            # Tampilkan pesan berdasarkan hasil prediksi
            if predicted_label == 'Normal':
                st.write("Berdasarkan hasil deteksi yang telah dilakukan berdasarkan gambar CT-Scan ginjal, anda dinyatakan NORMAL. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami.")                
            elif predicted_label == 'Batu Ginjal':
                st.write("Berdasarkan hasil deteksi yang telah dilakukan berdasarkan gambar CT-Scan ginjal, anda terdeteksi adanya BATU GINJAL. Segera konsultasi ke dokter spesialis untuk penanganan lebih lanjut.")                
            
