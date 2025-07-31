import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

#Load the pretrained model
scaler=joblib.load('Spotify_scaler.pkl')
model=load_model('wine_model.keras')

st.title('Wine Type Prediction')
st.write('This model predicts the type of wine based on various input features.Please enter the features to predict the type of your wine')

feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines','proline'
]
# Create input fields with unique keys
inputs = [
    st.slider("alcohol", min_value=11.0, max_value=15.0, value=13.0, step=0.1, key="input_alcohol"),
    st.slider("malic_acid", min_value=0.7, max_value=5.8, value=2.3, step=0.1, key="input_malic_acid"),
    st.slider("ash", min_value=1.3, max_value=3.3, value=2.4, step=0.1, key="input_ash"),
    st.slider("alcalinity_of_ash", min_value=10.0, max_value=30.0, value=19.5, step=0.1, key="input_alcalinity_of_ash"),
    st.slider("magnesium", min_value=70.0, max_value=165.0, value=99.0, step=1.0, key="input_magnesium"),
    st.slider("total_phenols", min_value=0.9, max_value=3.9, value=2.3, step=0.1, key="input_total_phenols"),
    st.slider("flavanoids", min_value=0.3, max_value=5.0, value=2.0, step=0.1, key="input_flavanoids"),
    st.slider("nonflavanoid_phenols", min_value=0.1, max_value=1.0, value=0.3, step=0.05, key="input_nonflavanoid_phenols"),
    st.slider("proanthocyanins", min_value=0.3, max_value=4.0, value=1.6, step=0.1, key="input_proanthocyanins"),
    st.slider("color_intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1, key="input_color_intensity"),
    st.slider("hue", min_value=0.4, max_value=1.7, value=1.0, step=0.05, key="input_hue"),
    st.slider("od280/od315_of_diluted_wines", min_value=1.2, max_value=4.0, value=3.0, step=0.1, key="input_od280/od315_of_diluted_wines"),
    st.slider("proline", min_value=270.0, max_value=1700.0, value=746.0, step=10.0, key="input_proline")
]


# Prediction button
if st.button('Predict'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([inputs], columns=feature_names)

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display the prediction
    st.success(f'🍇 Predicted Wine Quality: *{np.argmax(prediction) + 1}*')  # Assuming quality is 1-10