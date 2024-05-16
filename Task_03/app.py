import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    model_pkl_file = "Parkinsons_Disease_model.pkl"  
    with open(model_pkl_file, 'rb') as file:  
        return pickle.load(file)

model = load_data()

# Function to process comma-separated input
def process_input(input_string):
    # Split the string based on comma and strip any leading/trailing whitespace
    items = [item.strip() for item in input_string.split(',')]
    return items

# Main app
st.title('Comma-separated Input Processor')

# User input
user_input = st.text_input('Enter comma-separated values')

# Button to process the input
if st.button('Process'):
    if user_input:
        processed_data = pd.DataFrame(process_input(user_input))
        st.write('Processed Data:', processed_data)
    else:
        st.write('Please enter some data.')

if st.button('Predict'):
    # Organizing input features into the appropriate structure, e.g., dataframe
    input_data = pd.DataFrame(process_input(user_input))
    input_data = np.array(input_data).reshape(1, 22)
    # Predict
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    # Display prediction
    st.subheader('Prediction')
    st.write("Predicted Class:", prediction[0])
    st.write("Predicted Probability:", prediction_proba)

# Define the UI
st.title('Parkinson\'s Disease Predictor')
st.sidebar.header('User Input Features')

# Collect user input features
def collect_input_features():
    # Load the dataset
    parkinsons_df = pd.read_csv(r'C:\Users\Admin\Desktop\CodClause\Task_03\parkinsons.data')
    st.write(parkinsons_df.head(10))
    input_features = {}
    input_features['MDVP:Fo(Hz)'] = st.sidebar.number_input('MDVP:Fo(Hz)', min_value=40.0, max_value=350.0, value=150.0)
    input_features['MDVP:Fhi(Hz)'] = st.sidebar.number_input('MDVP:Fhi(Hz)', min_value=50.0, max_value=600.0, value=200.0)
    input_features['MDVP:Flo(Hz)'] = st.sidebar.number_input('MDVP:Flo(Hz)', min_value=40.0, max_value=500.0, value=100.0)
    input_features['MDVP:Jitter(%)'] = st.sidebar.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=2.0, value=0.5)
    input_features['MDVP:Jitter(Abs)'] = st.sidebar.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.1, value=0.01)
    input_features['MDVP:RAP'] = st.sidebar.number_input('MDVP:RAP', min_value=0.0, max_value=2.0, value=0.5)
    input_features['MDVP:PPQ'] = st.sidebar.number_input('MDVP:PPQ', min_value=0.0, max_value=2.0, value=0.5)
    input_features['Jitter:DDP'] = st.sidebar.number_input('Jitter:DDP', min_value=0.0, max_value=2.0, value=0.5)
    input_features['MDVP:Shimmer'] = st.sidebar.number_input('MDVP:Shimmer', min_value=0.0, max_value=2.0, value=0.5)
    input_features['MDVP:Shimmer(dB)'] = st.sidebar.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=20.0, value=5.0)
    input_features['Shimmer:APQ3'] = st.sidebar.number_input('Shimmer:APQ3', min_value=0.0, max_value=2.0, value=0.5)
    input_features['Shimmer:APQ5'] = st.sidebar.number_input('Shimmer:APQ5', min_value=0.0, max_value=2.0, value=0.5)
    input_features['MDVP:APQ'] = st.sidebar.number_input('MDVP:APQ', min_value=0.0, max_value=2.0, value=0.5)
    input_features['Shimmer:DDA'] = st.sidebar.number_input('Shimmer:DDA', min_value=0.0, max_value=2.0, value=0.5)
    input_features['NHR'] = st.sidebar.number_input('NHR', min_value=0.0, max_value=2.0, value=0.5)
    input_features['HNR'] = st.sidebar.number_input('HNR', min_value=0.0, max_value=35.0, value=10.0)
    input_features['RPDE'] = st.sidebar.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5)
    input_features['DFA'] = st.sidebar.number_input('DFA', min_value=0.0, max_value=1.0, value=0.5)
    input_features['spread1'] = st.sidebar.number_input('spread1', min_value=-10.0, max_value=10.0, value=0.0)
    input_features['spread2'] = st.sidebar.number_input('spread2', min_value=-10.0, max_value=10.0, value=0.0)
    input_features['D2'] = st.sidebar.number_input('D2', min_value=0.0, max_value=10.0, value=5.0)
    input_features['PPE'] = st.sidebar.number_input('PPE', min_value=0.0, max_value=1.0, value=0.5)
    return pd.DataFrame([input_features])

input_df = collect_input_features()

# Display the user input features
st.subheader('User Input features')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write("Predicted Class:", prediction[0])
st.write("Predicted Probability:", prediction_proba[0][1])
