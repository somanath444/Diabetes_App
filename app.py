import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step1: Load the pickled/trained model
model = open('rfc.pickle', 'rb') # rb - read binary format;
clf = pickle.load(model)
model.close()

# Step2: Get the user input from front end
pregs = st.number_input('Pregnancies',0, 20, step=1) 
glucose = st.slider('Glucose',40, 200, 40) 
bp = st.slider('BloodPressure',20, 140, 20) 
skin = st.slider('SkinThickness',7, 99, 7)
insulin = st.slider('Insulin',14, 850, 14)
bmi = st.slider('BMI',18, 70, 18) 
dpf = st.slider('DiabetesPedigreeFunction',0.05, 2.5, 0.05) 
age = st.slider('Age',21, 90, 21)
# for the sake of pridiction we dont need import
#the .py file doesnt recognize the above code so we need to import again

# step3: convert user input to model input
data = {
    'Pregnancies':pregs,
    'Glucose':glucose,
    'BloodPressure':bp,
    'SkinThickness':skin,
    'Insulin':insulin,
    'BMI':bmi,
    'DiabetesPedigreeFunction':dpf, 
    'Age':age
}
input_data = pd.DataFrame([data])

# step4 : get the predictions and print the result
prediction = clf.predict(input_data)[0]
if st.button("Predict"):
    if prediction==0:
        st.write("The Person is Healthy")
    if prediction==1:
        st.write("The Person has Diabetes")
