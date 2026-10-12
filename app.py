import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for shadow and card layout
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'knn.pkl' not found. Please place it in the same directory as app.py.")
    st.stop()

# App Header
st.title("🎓 Student Performance Predictor")
st.markdown("Estimate student performance scores using a K-Nearest Neighbors (KNN) Regressor model.")

# Input Form inside a Shadowed Container Layout
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Student Details Input")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age[cite: 1]", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender[cite: 1]", ["Male", "Female", "Other"])
    course = st.selectbox("Course[cite: 1]", ["Science", "Arts", "Commerce", "Engineering", "Medicine"])
    study_hours = st.number_input("Study Hours (per day)[cite: 1]", min_value=0.0, max_value=24.0, value=4.0)
    class_attendance = st.slider("Class Attendance (%)[cite: 1]", min_value=0, max_value=100, value=85)
    internet_access = st.selectbox("Internet Access[cite: 1]", ["Yes", "No"])

with col2:
    sleep_hours = st.number_input("Sleep Hours (per night)[cite: 1]", min_value=0.0, max_value=24.0, value=7.0)
    sleep_quality = st.selectbox("Sleep Quality[cite: 1]", ["Poor", "Average", "Good", "Excellent"])
    study_method = st.selectbox("Study Method[cite: 1]", ["Self-study", "Group study", "Online courses", "Tuition"])
    facility_rating = st.slider("Facility Rating (1-10)[cite: 1]", min_value=1, max_value=10, value=7)
    exam_difficulty = st.selectbox("Exam Difficulty[cite: 1]", ["Easy", "Medium", "Hard"])

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if st.button("Predict Performance"):
    # Note: Ensure that categorical variables (gender, course, etc.) 
    # are encoded identically to how they were transformed during model training 
    # (e.g., via OneHotEncoder or LabelEncoder).
    
    st.success("Model loaded successfully! Connect your preprocessing pipeline here to output the prediction.")
