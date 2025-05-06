#Library
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

#Loading models
preprocessor = pickle.load('preprocessor.pkl')
model = load_model('model.h5')

#Streamlit
st.write('Predictive model for Churn or not ')

CreditScore = st.number_input('CreditScore',min_value = 350, max_value = 850)
Geography = st.selectbox('Geography',options = ['France','Spain','Germany'])
Gender = st.selectbox('Gender',options = ['Male','Female'])
Age = st.slider('Age',18,92,25)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input('Balance',min_value = 0,max_value = 260000)
NumOfProducts = st.slider('NumOfProducts',1,4)
HasCrCard = st.selectbox('HasCrCard',options = [0,1])
IsActiveMember = st.selectbox('IsActiveMember',options = [0,1])
EstimatedSalary = st.number_input('EstimatedSalary',min_value = 11, max_value = 300000)


input_data = pd.DataFrame([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]],
                          columns = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'])
pre = preprocessor.transform(input_data)
pred = model.predict(pre)
pred_prob = pred[0][0]
st.write(f'Prediction: {pred_prob:.2f}')

if pred_prob>0.5:
    st.write('Customer willing to Churn')
else:
    st.write('Customer willing to Continue')