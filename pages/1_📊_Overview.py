import streamlit as st
from utils.preprocessing import load_data, preprocess

st.header("📊 Data Overview")

train, test, gender = load_data()
cleaned_train = preprocess(train)

st.subheader("🧼 Raw vs. Cleaned Training Data")
col1, col2 = st.columns(2)
col1.write("### Raw Data")
col1.dataframe(train.head())
col2.write("### After Preprocessing")
col2.dataframe(cleaned_train.head())

st.markdown("""
- **Missing values** in Age and Fare are filled with median.
- **Categorical columns** like Sex and Embarked are encoded numerically.
- **Irrelevant columns** (Cabin, Ticket, Name) are removed.
""")