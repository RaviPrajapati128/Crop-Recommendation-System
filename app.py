import numpy as np
import pandas as pd
import streamlit as st

st.markdown(
    "<h2 style='color:green; text-align:center;'>  🌾Data-Driven Crop Prediction Model  </h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h6 style='color:gray; text-align:center;'>Data-Driven Crop Prediction Based on NPK and Environmental Parameters</h6>", 
    unsafe_allow_html=True
)
st.divider()

data = pd.read_csv("Crop_recommendation.csv")

x = data.drop("label",axis = 1)
y = data["label"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

col1 , col2 = st.columns(2)

with col1:
    with st.expander("🌱 Soil & Nutrient Factors"):
        N = st.number_input("Enter ratio of Nitrogen content in soil",min_value = 0 ,max_value = 150 , value = 90)
        P = st.number_input("Enter ratio of Phosphorous content in soil",min_value = 0 ,max_value = 150 , value = 42)
        K = st.number_input("Enter ratio of Potassium content in soil",min_value = 0 ,max_value = 250 , value = 43)
        ph = st.slider("Enter ph value of the soil",0.0 ,14.0 ,6.5)

with col2:
    with st.expander("🌤️ Climate & Environmental Factors"):
        t = st.slider("Enter Temperature(degree Celsius)",0.0 ,50.0, 20.0)
        H = st.slider("Enter Relative humidity (%)",0.0 ,100.0 ,82.0)
        rainfall = st.number_input("Enter Rainfall(in mm)",min_value = 0.0 ,max_value = 300.0 ,value = 203.0)
        
user_input = np.array([[N,P,K,t,H,ph,rainfall]])

pred = knn.predict(user_input)

if st.button("Predict Crop",use_container_width=True):
    with st.container():
        st.markdown("### 🌱 Crop Recommendation")
        st.markdown(
            f"Based on soil and nutrient factors, as well as climate and environmental conditions, "
            f"I recommend growing **:green[{pred[0]}]** 🌾"
        )

st.divider()

with st.expander("ℹ️ About the Project"):
    st.markdown("""
    ### 🌾 Crop Recommendation System

    This project uses **machine learning** to suggest the best crop 
    based on soil nutrients (N, P, K, pH) and climate conditions 
    (temperature, humidity, rainfall).

    **Purpose**
    - 📊 Help farmers and students make data-driven decisions  
    - 🌱 Support sustainable farming practices  
    - 🖥️ Showcase an interactive dashboard built with Streamlit  

    **Outcome**
    - Provides simple, clear crop recommendations  
    - Encourages efficient use of resources  
    """)

