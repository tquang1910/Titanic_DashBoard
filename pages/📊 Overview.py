
import streamlit as st
from utils.preprocessing import load_data, preprocess
import pandas as pd

st.header("📊 Titanic Dataset Overview (Enhanced)")

# Load and preprocess data
train, test, gender = load_data()
cleaned_train = preprocess(train)

# Summary Statistics
st.subheader("📄 Dataset Summary")
total_passengers = len(train)
missing_age = train['Age'].isnull().sum()
missing_fare = train['Fare'].isnull().sum()
missing_embarked = train['Embarked'].isnull().sum()

summary_dict = {
    "Total Passengers": total_passengers,
    "Missing Age": missing_age,
    "Missing Fare": missing_fare,
    "Missing Embarked": missing_embarked,
}

summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['Count'])
st.table(summary_df)

st.markdown("---")

# Display Raw vs Cleaned side-by-side
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🗂 Raw Data (Top 10 Rows)")
    st.dataframe(train.head(10), use_container_width=True)
with col2:
    st.markdown("### ✨ Cleaned Data (Top 10 Rows)")
    st.dataframe(cleaned_train.head(10), use_container_width=True)

st.markdown("""
✅ **Cleaning steps applied:**
- Missing Age & Fare filled by median value
- Categorical columns (Sex, Embarked) encoded numerically
- Irrelevant columns removed: Cabin, Ticket, Name
""")
