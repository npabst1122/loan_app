import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import os

# Fix for deprecated numpy types (helps with pickle compatibility)
np.bool = np.bool_
np.object = object

# Helper functions for encoding
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, 0)

def get_value(val, my_dict):
    return my_dict.get(val, 0)

# Sidebar page selector
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

# ---------------- Home Page ---------------- #
if app_mode == 'Home':
    st.title('🏦 LOAN PREDICTION APP')
    
    # Image display
    if os.path.exists('loan_image.jpg'):
        st.image('loan_image.jpg', use_container_width=True)
    else:
        st.warning("Home image (loan_image.jpg) not found.")
    
    st.write('🔍 @DSU — for learning purposes only.')

    # CSV preview
    try:
        csv = pd.read_csv("test.csv")
        st.subheader("📄 Dataset Preview")
        st.write(csv.head())
    except FileNotFoundError:
        st.warning("test.csv not found. Please upload it to preview.")


# ---------------- Prediction Page ---------------- #
elif app_mode == 'Prediction':
    st.subheader('📋 Fill out the form to receive your loan eligibility result.')

    st.sidebar.header("📌 Client Information")

    # Dictionaries for mapping
    gender_dict = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu = {"Graduate": 1, "Not Graduate": 2}
    prop = {"Rural": 1, "Urban": 2, "Semiurban": 3}

    # User input
    Gender = st.sidebar.radio('Gender', list(gender_dict.keys()))
    Married = st.sidebar.radio('Married', list(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed', list(feature_dict.keys()))
    Dependents = st.sidebar.radio('Dependents', options=['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', list(edu.keys()))
    ApplicantIncome = st.sidebar.slider('Applicant Income', 0, 10000, 500)
    CoapplicantIncome = st.sidebar.slider('Coapplicant Income', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('Loan Amount in K$', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan Term (Months)', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit History', (0.0, 1.0))
    Property_Area = st.sidebar.radio('Property Area', list(prop.keys()))

    # One-hot encode Dependents
    class_0 = class_1 = class_2 = class_3 = 0
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2':
        class_2 = 1
    else:
        class_3 = 1

    # One-hot encode Property Area
    Rural = Urban = Semiurban = 0
    if Property_Area == 'Urban':
        Urban = 1
    elif Property_Area == 'Semiurban':
        Semiurban = 1
    else:
        Rural = 1

    # Final input vector for model
    feature_list = [
        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
        Credit_History, get_value(Gender, gender_dict),
        get_fvalue(Married),
        class_0, class_1, class_2, class_3,
        get_value(Education, edu),
        get_fvalue(Self_Employed),
        Rural, Urban, Semiurban
    ]

    single_sample = np.array(feature_list).reshape(1, -1)

    # Prediction button
    if st.button("🔮 Predict"):
        try:
            # Load model
            loaded_model = pickle.load(open('RF.sav', 'rb'))

            # Load GIFs
            success_gif = base64.b64encode(open("6m-rain.gif", "rb").read()).decode("utf-8")
            fail_gif = base64.b64encode(open("green-cola-no.gif", "rb").read()).decode("utf-8")

            # Run prediction
            prediction = loaded_model.predict(single_sample)

            # Output results
            if prediction[0] == 1:
                st.success("✅ Congratulations! You will get the loan.")
                st.markdown(f'<img src="data:image/gif;base64,{success_gif}" alt="success gif">', unsafe_allow_html=True)
            else:
                st.error("❌ Unfortunately, you will not get the loan.")
                st.markdown(f'<img src="data:image/gif;base64,{fail_gif}" alt="failure gif">', unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error("🚫 Required file missing. Make sure RF.sav and GIF files are in the app directory.")
