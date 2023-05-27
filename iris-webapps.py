import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

st.write("""
# Web Apps - Klasifikasi Bunga Iris
Aplikasi Berbasis Web untuk Mengklasifikasi **BUNGA IRIS**!
""")

# img = Image.open("img.jpg")
# st.image(img, use_column_width = False)

st.sidebar.header("Parameter Inputan")

def input_user():
    panjang_sepal = st.sidebar.slider("Panjang Sepal", 0.1, 10.1, 3.1)
    lebar_sepal = st.sidebar.slider("Lebar Sepal", 0.1, 10.1, 3.8)
    panjang_petal = st.sidebar.slider("Panjang Petal", 0.1, 10.1, 3.1)
    lebar_petal = st.sidebar.slider("Lebar Petal", 0.1, 10.1, 3.8)

    data = {"panjang sepal" : panjang_sepal,
            "lebar sepal" : lebar_sepal,
            "panjang petal" : panjang_petal,
            "lebar petal" : lebar_petal}
    fitur = pd.DataFrame(data, index = [0])

    return fitur

df = input_user()

st.subheader("Parameter Input")
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

model = GaussianNB()
model.fit(X, Y)

prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

st.subheader("Label Kelas dan Nomor Indeks yang Sesuai")
st.write(iris.target_names)

st.subheader("Hasil Prediksi")
st.write(iris.target_names[prediksi])

st.subheader("Probabilitas Hasil Prediksi")
st.write(prediksi_proba)