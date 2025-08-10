import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



st.title('Transportni klassifikatsiya qiluvchi model')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    img = PILImage.create(file)
    model = load_learner('Transpoer_model.pkl')

    pred, pred_id, probs = model.predict(img)

    st.image(img, width=170)

    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=model.dls.vocab, y=probs * 100)
    st.plotly_chart(fig)

