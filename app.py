import numpy as np
from PIL import Image, ImageOps
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

st.set_option('deprecation.showfileUploaderEncoding', False)

model = model_from_json(open("vgg16_model.json", "r").read())
model.load_weights('vgg16_model_weights.h5')

st.write("""# Face Expression Recognition""")
st.write("This Model predicts the expression on a persons face.")
file = st.file_uploader("Please upload an image file", type=["jpg"])


def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_haar_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('ANGRY 😡', 'HAPPY 😀', 'NEUTRAL 😐', 'SAD 🙁')
        predicted = emotions[max_index]
    return predicted


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.text("The given image has the following expression:")
    st.write(prediction)
    st.text("*****Built By Data Science Community SRM with 💜 *****")
