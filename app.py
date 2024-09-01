# -*- coding: utf-8 -*-

import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('saved_model.h5')

# Définir la fonction de prédiction
def predict_image(img):
    img = image.load_img(img, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        return "Mycosphaerella"
    else:
        return "Dreschleria"

# Créer et lancer l'interface Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Text(label="Prediction"),
    
)

iface.launch()
