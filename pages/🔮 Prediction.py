
import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import load_data, preprocess
from sklearn.ensemble import RandomForestClassifier

st.header("🔮 Titanic Survival Prediction")

# Load and preprocess training data for model training
train, _, _ = load_data()
train = preprocess(train)

# Features and target
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train['Survived']

# Train a simple model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# User input
st.subheader("📝 Passenger Information")
pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)

# Encode input
sex_encoded = 0 if sex == "Male" else 1
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                           columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])

# Prediction
if st.button("🚀 Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Prediction: Survived ({probability*100:.1f}% confidence)")
    else:
        st.error(f"❌ Prediction: Did not survive ({probability*100:.1f}% confidence)")
