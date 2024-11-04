from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

class_names = iris.target_names

model = load_model('iris_model.keras')

#Configuración streamlit
st.title('clasificación de flores Iris')
st.write('Parámetros de entrada')

#Selección de características
petal_length = st.slider('Longitud del pétalo', 1.0, 7.0, 1.5)
sepal_length = st.slider('Longitud del sépalo', 4.0, 8.0, 5.0)
petal_width = st.slider('Ancho del pétalo', 0.1, 2.5, 0.2)
sepal_width = st.slider('Ancho del sépalo', 2.0, 4.5, 3.0)

#Botón para predecir
if st.button('Predecir'):
    #Arreglo con los datos
    datos = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    #Predecir la clase
    prediction= model.predict(datos)
    prediction_class = class_names[np.argmax(prediction)]
    st.write(f'La especie de flor es: {prediction_class}')