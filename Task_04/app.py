import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset (assuming you have a dataset named 'heart_disease.csv')
@st.cache_data
def load_data():
    return pd.read_csv('Heart_Disease_Prediction.csv')

# Preprocess the data
def preprocess_data(data):
    # Perform any necessary preprocessing here
    data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return data

# Train the Random Forest model
def train_model(data):
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test,y_pred))
    return model, X_test, y_test

# Predict using the trained model
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Heart Disease Risk Assessment')
    
    # Load data
    data = load_data()
    
    # Preprocess data
    data = preprocess_data(data)

    # Train model
    model, X_test, y_test = train_model(data)
    st.header("PreProcess Data")
    st.write(data)
    st.header("User Input Feature")

    # Sidebar for user input
    st.sidebar.header('Input Your Health Metrics')
    age = st.sidebar.number_input('Age', min_value=0, max_value=150, value=25)
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=0, max_value=600, value=200)
    fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dL', ['No', 'Yes'])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
    exang = st.sidebar.radio('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=0.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.sidebar.number_input('Number of Major Vessels (0-3) Colored by Fluoroscopy', min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])
    
    # Prepare input data for prediction
    sex_mapping = {'Male': 1, 'Female': 0}
    fbs_mapping = {'No': 0, 'Yes': 1}
    exang_mapping = {'No': 0, 'Yes': 1}
    input_data = np.array([[age, sex_mapping[sex], cp, trestbps, chol, fbs_mapping[fbs], restecg, thalach, exang_mapping[exang], oldpeak, slope, ca, thal]])
    input_df = pd.DataFrame(input_data)
    st.write(input_df)
    # Make prediction
    if st.sidebar.button('Predict'):
        prediction = predict(model, input_data)
        prediction_proba = model.predict_proba(input_data)
        if prediction[0] == 1:
            st.write('Prediction: High risk of heart disease')
            st.write("Predicted Probability:", prediction_proba[0][1])
        else:
            st.write('Prediction: Low risk of heart disease')
            st.write("Predicted Probability:", prediction_proba[0][1])

    if st.button('Prediction'):
        prediction = predict(model, input_data)
        prediction_proba = model.predict_proba(input_data)
        if prediction[0] == 1:
            st.write('Prediction: High risk of heart disease')
            st.write("Predicted Probability:", prediction_proba[0][1])
        else:
            st.write('Prediction: Low risk of heart disease')
            st.write("Predicted Probability:", prediction_proba[0][1])
if __name__ == '__main__':
    main()
