import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf

from prediction import get_prediction


st.set_page_config(page_title='Hostpital Patient Survival Prediction', page_icon="🏥", layout="wide", initial_sidebar_state='expanded')

model = load_model('keras_model.h5')

# creating option list for dropdown menu

features = ['apache_3j_diagnosis','gcs_motor_apache', 'd1_lactate_max', 'd1_lactate_min','apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
#'apache_3j_diagnosis', 'gcs_motor_apache', 'd1_lactate_max','d1_lactate_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob'], dtype='object'

st.markdown("<h1 style='text-align: center;'>Patient Survival Prediction App 🏥 </h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.header("Predict the input for following features:") 
                
        apache_3j_diagnosis = st.slider('gcs_motor_apache', 0.0300, 2201.05, value=1.0000, format="%f")  
        gcs_motor_apache = st.slider('gcs_motor_apache', 1.0000, 6.0000, value=1.0000, format="%f")
        d1_lactate_max = st.selectbox( 'd1_lactate_max:', [3.970, 5.990, 0.900, 1.100, 1.200, 2.927, 9.960, 19.500])
        d1_lactate_min = st.selectbox('d1_lactate_min:', [2.380, 2.680, 6.860, 0.900, 1.100, 1.200, 1.000, 2.125])
        apache_4a_hospital_death_prob = st.selectbox( 'apache_4a_hospital_death_prob:', [0.990, 0.980, 0.950, 0.040, 0.030, 0.086, 0.020, 0.010])
        apache_4a_icu_death_prob = st.selectbox('apache_4a_icu_death_prob:', [0.950, 0.940, 0.920, 0.030, 0.043, 0.030, 0.043, 0.020, 0.010])
        submit = st.form_submit_button("Predict")

    if submit:

        data= np.array([apache_3j_diagnosis, gcs_motor_apache, d1_lactate_max, d1_lactate_min, apache_4a_hospital_death_prob, apache_4a_icu_death_prob]).reshape(1, -1)
        
        pred = get_prediction(data=data, model=model)

        if pred[0][0]< 0.5:
            survival = 'No'
        elif pred[0][0] > 0.5:
            survival = 'Yes'

        
        
        st.write(f"The predicted Patient Survival is:  {survival}")


if __name__ == '__main__':
       main()
