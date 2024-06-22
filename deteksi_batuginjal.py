import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Define a function to preprocess the image and make predictions
def predict_kidney_stone(image):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def colored_divider(color, thickness="13px"):
    st.markdown(f"<hr style='border: none; border-top: {thickness} solid {color};' />", unsafe_allow_html=True)

def large_title(text, size="45px"): 
    st.markdown(f"<h1 style='font-size: {size};'>{text}</h1>", unsafe_allow_html=True)
def large_text(text, size="20px"):
    st.markdown(f"<p style='font-size: {size};'>{text}</p>", unsafe_allow_html=True)

# Streamlit interface
st.sidebar.title("Pilih Menu")
page = st.sidebar.selectbox("MENU", ["Beranda", "Penjelasan Batu Ginjal", "Deteksi"])

if page == "Beranda":
    large_title("APLIKASI DETEKSI BATU GINJAL")
    colored_divider(color="red")
    large_text("Aplikasi ini membantu deteksi BATU GINJAL berdasarkan hasil gambar CT-Scan.")

elif page == "Penjelasan Batu Ginjal":
    large_title("PENJELASAN BATU GINJAL")
    colored_divider(color="red")
    large_text("Batu ginjal adalah batu yang terbentuk di tubuli ginjal kemudian berada di kaliks, infundibulum, pelvis ginjal dan bahkan bisa mengisi pelvis serta seluruh kaliks ginjal dan merupakan batu saluran kemih yang paling sering terjadi. Penyebab terbentuknya batu saluran kemih diduga berhubungan dengan gangguan aliran urine, gangguan metabolik, infeksi saluran kemih.")

elif page == "Deteksi":
    large_title("Mulai Deteksi")

    uploaded_file = st.file_uploader("Masukkan gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Mendeteksi...")
    
        class_name, confidence_score = predict_kidney_stone(image)
    
        st.write(f"HASIL: {class_name}")
        st.write(f"NILAI AKURAT: {confidence_score:.2f}")
