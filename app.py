import streamlit as st 
from fastai.vision.all import * 
import plotly.express as px
st.title('transportni classification qiluvchi model')

# rasm joylash
file = st.file_uploader('Rasm yuklash' , type=['png','jpg'])
if file:
    st.image(file)
    img = PILImage.create(file)

    modal = load_learner('transport.pkl')
    pred, pred_id, probs = modal.predict(img)

    st.success(f'bashora {pred}')
    st.info(f'ehtimollik {probs[pred_id]}')

    fig = px.bar(x=probs*100, y=modal.dls.vocab)
    st.plotly_chart(fig)
