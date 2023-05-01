import requests
import streamlit as st

# Define function to make API call
def get_prediction(data):
    url = 'http://localhost:8000/predict/'
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return response.text

# Define function to get input from user
def get_input():
    cs = st.number_input('Credit Score (Min = 350 and Max = 850)', value=0, step=1)
    geo = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age - Years (Min = 18 and Max = 92)', value=0, step=1)
    tenure = st.selectbox('Tenure', [0,1,2,3,4,5,6,7,8,9,10])
    bal = st.number_input('Balance $ (Min = 0 and Max = 250898.09)', value=0.00, step=0.01)
    prod = st.selectbox('Number of Products', [1,2,3,4])
    crd = st.selectbox('Has Credit Card (1=Yes, 0=No)', [0,1])
    memb = st.selectbox('Is Active Member (1=Yes, 0=No)', [0,1])
    sal = st.number_input('Estimated Salary $ (Min = 11.58 and Max = 199992.48)', value=0.00, step=0.01)
    
    data = {
        "CreditScore": cs,
        "Geography": geo,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": bal,
        "NumOfProducts": prod,
        "HasCrCard": crd,
        "IsActiveMember": memb,
        "EstimatedSalary": sal
    }
    
    return data

# Define function to display prediction result
def show_prediction_result(result):
    st.write('Prediction Result: ', result)

# Set page title
st.set_page_config(page_title='Churn Prediction App')

# Add title and description
st.title('Churn Prediction App')
st.write('This app predicts customer churn using a machine learning model.')

# Get input from user
input_data = get_input()

# Make prediction and display result
if st.button('Predict'):
    prediction_result = get_prediction(input_data)
    show_prediction_result(prediction_result)