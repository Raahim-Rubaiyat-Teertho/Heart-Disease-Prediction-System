import streamlit as st
import pandas as pd
import pickle as p

st.title('Heart Disease Prediction System')

with st.form(key='form1'):
    # Age input
    age = st.number_input('Age', min_value=0, max_value=120)

    # Sex input
    sex = st.selectbox('Sex', options=['Male', 'Female'])

    # Chest pain type options (encoded as 0-3)
    chest_pain = st.selectbox(
        'Type of Chest Pain',
        options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
    )

    # Resting blood pressure input
    resting_blood_pressure = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=0)

    # Cholesterol level input
    cholesterol_level = st.number_input('Cholesterol Level (in mg/dl)', min_value=0)

    # Fasting blood sugar input
    fasting_blood_sugar = st.selectbox('Is Fasting Blood Sugar greater than 120 mg/dl?', options=['Yes', 'No'])

    # Resting electrocardiograph results input
    ecg_results = st.selectbox('Resting Electrocardiograph Results', options=[0, 1, 2])

    # Maximum heart rate input
    max_heart_rate = st.number_input('Maximum Achieved Heart Rate', min_value=0)

    # Exercise induced angina input
    exercise_angina = st.selectbox('Exercise Induced Angina?', options=['Yes', 'No'])

    # ST depression input
    st_depression = st.number_input('ST Depression induced by exercise relative to rest', min_value=0.0, format='%.1f')

    # Slope of the peak exercise ST segment input (using descriptive categories)
    slope_st_segment = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'])

    # Number of major vessels input
    major_vessels = st.selectbox('Number of Major Vessels (0-3) colored by fluoroscopy', options=[0, 1, 2, 3])

    # Thal input
    thal = st.selectbox('Thalassemia: 0 = normal; 1 = fixed defect; 2 = reversible defect', options=[0, 1, 2])

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Convert inputs to appropriate format for prediction
        sex_encoded = 1 if sex == 'Male' else 0
        chest_pain_encoded = {
            'Typical Angina': 0,
            'Atypical Angina': 1,
            'Non-Anginal Pain': 2,
            'Asymptomatic': 3
        }[chest_pain]
        fasting_blood_sugar_encoded = 1 if fasting_blood_sugar == 'Yes' else 0
        exercise_angina_encoded = 1 if exercise_angina == 'Yes' else 0

        # Encode slope of ST segment as per categories
        slope_st_segment_encoded = {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        }[slope_st_segment]

        # Prepare the feature array for prediction
        features = [
            age,
            sex_encoded,
            chest_pain_encoded,
            resting_blood_pressure,
            cholesterol_level,
            fasting_blood_sugar_encoded,
            ecg_results,
            max_heart_rate,
            exercise_angina_encoded,
            st_depression,
            slope_st_segment_encoded,  # Use the encoded slope here
            major_vessels,
            thal
        ]

        # Make prediction
        with open('./heart_disease_prediction_model.pkl', 'rb') as f:
            model = p.load(f)
            prediction = model.predict_proba([features])

        # Display the prediction
        st.subheader(f'You have a {prediction[0][1] * 100:.2f}% chance of having heart disease.')
