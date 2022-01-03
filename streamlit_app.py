import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile


# st.title("Game Classifier")
st.set_page_config(page_title="Game Classifier", layout="wide")
st.header("Please input a screenshot of AC Unity or Hitman")
st.text("Created by PNH")


model = None
img_height = 180
img_width = 180
class_names = ['AC Unity', 'Hitman']

def load_and_predict(image):
    
    img = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img = img.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    global model
    if model is None:
        model = keras.models.load_model("gameClass.h5")

    predictions = model.predict(img_array)
    score = 100 * np.max(tf.nn.softmax(predictions[0]))
    pred = class_names[np.argmax(predictions[0])]

    return pred, score
    

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])
    temp_file = NamedTemporaryFile(delete=False)
    if uploaded_file is not None:
        temp_file.write(uploaded_file.getvalue())
    btn = st.button("Predict")
    if btn:
        if uploaded_file is None:
            st.write("No image selected")
        else:
            image = Image.open(uploaded_file)
            predictions, score = load_and_predict(temp_file.name)
            st.success("Classification Complete!")
            st.write(f"The model is {round(score, 2)}% confident that it belongs to {predictions}")
            st.image(image, caption=f"{predictions}", use_column_width=True)


