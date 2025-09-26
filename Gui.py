import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model bundle
@st.cache_resource
def load_model():
    with st.spinner('Loading model components...'):
        try:
            with open("heart_model_xgbb.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error("Error: Model file 'heart_model_xgbb.pkl' not found.")
            st.info("Please make sure the file is in the same directory as this script and you have run the notebook cell to save the model.")
            return None
        except Exception as e:
            st.error(f"Error loading model bundle: {e}")
            return None

# Main app
def main():
    st.title("Heart Disease Prediction App 🩺")
    st.write("Enter the patient details below to predict the likelihood of heart disease.")

    bundle = load_model()
    if bundle is None:
        st.stop()

    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    numeric_cols = bundle["numeric_cols"]
    
    st.markdown("### Patient Information")
    
    # Input fields in the correct order to match the model's features
    input_data = {}
    input_data['age'] = st.number_input("Age", min_value=0, max_value=120, value=50, help="Age in years.")
    input_data['sex'] = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], help="Gender of the patient.")[1]
    input_data['cp'] = st.selectbox("Chest Pain Type", options=[("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal Pain", 3), ("Asymptomatic", 0)], format_func=lambda x: x[0], help="The type of chest pain experienced.")[1]
    input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120, help="The patient's resting blood pressure.")
    input_data['chol'] = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, help="The patient's serum cholesterol level.")
    input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Fasting blood sugar level.")[1]
    input_data['restecg'] = st.selectbox("Resting ECG Results", options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0], help="Results of the resting electrocardiogram.")[1]
    input_data['thalach'] = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150, help="The highest heart rate achieved during exercise.")
    input_data['exang'] = st.selectbox("Exercise Induced Angina", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Chest pain caused by exercise.")[1]
    input_data['oldpeak'] = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    input_data['slope'] = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 1.0), ("Flat", 2.0), ("Downsloping", 0.0)], format_func=lambda x: x[0], help="The slope of the peak exercise ST segment.")[1]
    input_data['ca'] = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, help="Number of major vessels (0-3) colored by fluoroscopy.")
    input_data['thal'] = st.selectbox("Thalassemia", options=[("Normal", 2), ("Fixed Defect", 1), ("Reversible Defect", 3)], format_func=lambda x: x[0], help="A type of blood disorder called thalassemia.")[1]
    
    # Create DataFrame and ensure column order matches the model
    input_df = pd.DataFrame([input_data])[features]
    
    # Scale numeric columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Make prediction
    if st.button("Predict"):
        with st.spinner('Generating Prediction...'):
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]

                st.markdown("### Prediction Results")
                if prediction == 1:
                    st.error("The model predicts a **HIGH likelihood** of heart disease.")
                else:
                    st.success("The model predicts a **LOW likelihood** of heart disease.")
                
                st.write(f"Probability of Heart Disease: **{probability[1]:.2%}**")
                st.write(f"Probability of No Heart Disease: **{probability[0]:.2%}**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
