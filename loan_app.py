import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Functions to convert categorical inputs
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    return feature_dict.get(val, 0)

def get_value(val, my_dict):
    return my_dict.get(val, 0)

# Sidebar page selector
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

# HOME PAGE
if app_mode == 'Home':
    st.title('🏦 LOAN PREDICTION APP')
    st.image('loan_image.jpg', use_column_width=True)
    st.markdown('### Dataset Preview:')
    data = pd.read_csv('test.csv')
    st.write(data.head())

    st.markdown('### Applicant Income vs Loan Amount')
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20))

# PREDICTION PAGE
elif app_mode == 'Prediction':
    st.subheader('💼 Please fill in the client information:')

    # Sidebar input fields
    gender_dict = {"Male":1, "Female":2}
    edu = {'Graduate':1, 'Not Graduate':2}
    prop = {'Rural':1, 'Urban':2, 'Semiurban':3}

    st.sidebar.header("Client Information")
    ApplicantIncome = st.sidebar.slider('Applicant Income', 0, 10000, 500)
    CoapplicantIncome = st.sidebar.slider('Coapplicant Income', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('Loan Amount (in K$)', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan Term (months)', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit History', (0.0, 1.0))
    Gender = st.sidebar.radio('Gender', list(gender_dict.keys()))
    Married = st.sidebar.radio('Married', ['Yes', 'No'])
    Self_Employed = st.sidebar.radio('Self Employed', ['Yes', 'No'])
    Dependents = st.sidebar.radio('Dependents', ['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', list(edu.keys()))
    Property_Area = st.sidebar.radio('Property Area', list(prop.keys()))

    # One-Hot Encoding for Dependents
    class_0 = class_1 = class_2 = class_3 = 0
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2':
        class_2 = 1
    else:
        class_3 = 1

    # One-Hot Encoding for Property_Area
    Rural = Urban = Semiurban = 0
    if Property_Area == 'Rural':
        Rural = 1
    elif Property_Area == 'Urban':
        Urban = 1
    else:
        Semiurban = 1

    # Final input vector
    feature_list = [
        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
        Credit_History,
        get_value(Gender, gender_dict),
        get_fvalue(Married),
        class_0, class_1, class_2, class_3,
        get_value(Education, edu),
        get_fvalue(Self_Employed),
        Rural, Urban, Semiurban
    ]
    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button("Predict"):
        file_ = open("6m-rain.gif", "rb")
        success_gif = base64.b64encode(file_.read()).decode("utf-8")

        file = open("green-cola-no.gif", "rb")
        fail_gif = base64.b64encode(file.read()).decode("utf-8")

        model = pickle.load(open("RF.sav", "rb"))
        prediction = model.predict(single_sample)

        if prediction[0] == 0:
            st.error("❌ Unfortunately, you will not get the loan.")
            st.markdown(f'<img src="data:image/gif;base64,{fail_gif}" alt="failure gif">', unsafe_allow_html=True)
        else:
            st.success("✅ Congratulations! You will get the loan.")
            st.markdown(f'<img src="data:image/gif;base64,{success_gif}" alt="success gif">', unsafe_allow_html=True)
