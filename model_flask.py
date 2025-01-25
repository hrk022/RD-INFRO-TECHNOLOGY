import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
with open('churn_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

with open('churn_predictionScaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Display encoding information
st.write("### Categorical Encodings")
st.write("""
- **Geography**:  
  - Germany: 0  
  - Spain: 1  
  - France: 2  
- **Has Credit Card**:  
  - Yes: 1  
  - No: 0  
- **Is Active Member**:  
  - Yes: 1  
  - No: 0  
""")

# Input fields
CreditScore = st.number_input("CreditScore", min_value=350.0, max_value=850.0, value=400.0)
Geography = st.number_input("Geography", min_value=0, max_value=3, value=2)  # Adjusted to match encoding
Age = st.number_input("Age", min_value=18, max_value=67, value=40)
Balance = st.number_input("Balance", min_value=0.0, max_value=250898.09, value=4.0)
NumOfProducts = st.number_input("NumOfProducts", min_value=1, max_value=4, value=4)
HasCrCard = st.number_input("HasCrCard", min_value=0, max_value=1, value=1)
IsActiveMember = st.number_input("IsActiveMember", min_value=0, max_value=1, value=1)
EstimatedSalary = st.number_input("EstimatedSalary", min_value=0.0, max_value=200000.0, value=40.0)

# Prepare input data
input_data = np.array([[CreditScore, Geography, Age, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
input_scaler = scaler.transform(input_data)
# Prediction
if st.button("Predict Exited"):
    prediction = model.predict(input_scaler)
    result = "Exited" if prediction[0] == 1 else "Not Exited"
    st.write(f"Prediction: {result}")
