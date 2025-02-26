import streamlit as st
import numpy as np
import pickle

# Load Encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load Trained Model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to preprocess input
def preprocess_input(user_input, encoders):
    processed_input = []
    for col, value in user_input.items():
        if col in encoders:
            processed_input.append(encoders[col].transform([value])[0])
        else:
            processed_input.append(value)
    return np.array(processed_input).reshape(1, -1)

# Function to predict autism
def predict_autism(user_input):
    input_data = preprocess_input(user_input, encoders)
    prediction = model.predict(input_data)
    return "🟢 No Autism Detected" if prediction[0] == 0 else "🔴 Autism Detected"

st.markdown(
    """
    <style>
         /* Set background color */
        [data-testid="stAppViewContainer"] {
            background-color: #121212;
        }
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }
        /* Text color */
        .stTextInput, .stNumberInput, .stSelectbox, .stRadio, .stSlider {
            color: white;
        }
        .medium-font {
            font-size: 38px;
            font-weight: bold;
            color:yellow;
        }
        .custom-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            
        }
        label {
            color: white !important;
            
        }
        
        /* Style the form container with white border */
        div[data-testid="stForm"] {
            border: 2px solid white !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin-top: 20px;
            
        }

        /* Explicitly set radio, selectbox, slider, and number input option colors */
    div[data-testid="stRadio"] * {
        color: white !important;
    }

    div[data-testid="stSlider"] * {
        color: white !important;
    }

    .center-button {
        display: flex;
        justify-content: center;
    }

        div.stButton > button {
        width: 100% !important;  /* Make button fill its container */
        font-size: 18px !important;  /* Increase font size */
        border-radius: 8px !important; /* Add rounded corners */
        font-weight: bold !important; /* Make text bold */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='medium-font'>🧠 Autism Spectrum Disorder Prediction</p>", unsafe_allow_html=True)
st.write("<p class='stTextInput'>Fill in the details below to check the ASD likelihood</p>", unsafe_allow_html=True)

# User Input Form
with st.form("autism_form"):
    # A1 to A5 in first row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        A1_Score = st.radio("A1 Score", [0, 1])
    with col2:
        A2_Score = st.radio("A2 Score", [0, 1])
    with col3:
        A3_Score = st.radio("A3 Score", [0, 1])
    with col4:
        A4_Score = st.radio("A4 Score", [0, 1])
    with col5:
        A5_Score = st.radio("A5 Score", [0, 1])

    # A6 to A10 in second row
    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        A6_Score = st.radio("A6 Score", [0, 1])
    with col7:
        A7_Score = st.radio("A7 Score", [0, 1])
    with col8:
        A8_Score = st.radio("A8 Score", [0, 1])
    with col9:
        A9_Score = st.radio("A9 Score", [0, 1])
    with col10:
        A10_Score = st.radio("A10 Score", [0, 1])

    # Age input
    age = st.number_input("Age", min_value=1, max_value=100, value=25)

    # Country of Residence
    country_list = sorted(["Afghanistan", "India", "United States", "United Kingdom", "Japan", "China", "France", "Germany", "Australia", "Canada"])
    contry_of_res = st.selectbox("Country of Residence", country_list)

    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["m", "f"])
    with col2:
        jaundice = st.radio("Had Jaundice?", ["yes", "no"])

    col3, col4 = st.columns(2)
    with col3:
        austim = st.radio("Family Autism History?", ["yes", "no"])
    with col4:
        used_app_before = st.radio("Used Screening App Before?", ["yes", "no"])
        
    # Other inputs
    ethnicity = st.selectbox("Ethnicity", ["Others", "White-European", "Middle Eastern", "Asian", "Black", "Hispanic"])
    result = st.slider("Screening Test Score", 0.0, 20.0, 4.5)
    relation = st.selectbox("Who is filling the form?", ["Self", "Others"])
    
    # Center the Predict button using st.columns
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 2, 1, 1, 1])  # Adjust column width ratios

    with col4:  # Place button in the center column
        submit = st.form_submit_button("🔍 Predict")

# On Submit, Predict Autism
if submit:
    user_input = {
        "A1_Score": A1_Score, "A2_Score": A2_Score, "A3_Score": A3_Score,
        "A4_Score": A4_Score, "A5_Score": A5_Score, "A6_Score": A6_Score,
        "A7_Score": A7_Score, "A8_Score": A8_Score, "A9_Score": A9_Score,
        "A10_Score": A10_Score, "age": age, "gender": gender, "ethnicity": ethnicity,
        "jaundice": jaundice, "austim": austim, "contry_of_res": contry_of_res,
        "used_app_before": used_app_before, "result": result, "relation": relation
    }
    prediction = predict_autism(user_input)
    st.success(prediction)