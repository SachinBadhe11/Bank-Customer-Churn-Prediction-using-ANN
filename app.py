import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pickle
from tensorflow.keras.models import load_model

#load pickle file
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('OneHotEncoder_Geography.pkl', 'rb') as f:
    OneHotEncoder_Geography = pickle.load(f)

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)




from re import escape
#streamlit app
st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography', OneHotEncoder_Geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=0, max_value=120, step=1)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure',0,10)
num_of_products = st.number_input('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Geography': [geography],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})


#one-hot encode geography
Geography_encoded = OneHotEncoder_Geography.transform(input_data[['Geography']]).toarray()
Geography_encoded_df = pd.DataFrame(Geography_encoded, columns=OneHotEncoder_Geography.get_feature_names_out(['Geography']))

#combine columns
input_data_df = pd.concat([input_data, Geography_encoded_df], axis=1)

#concatination with OHE data
input_data_df=input_data_df.drop('Geography',axis=1)

#Scaling input data
input_data_scaled=scaler.transform(input_data_df)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]
if prediction_prob>0.5:
    st.write('Customer will churn')
else:
    st.write('Customer will not churn')





