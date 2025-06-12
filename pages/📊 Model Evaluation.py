
import streamlit as st
from utils.preprocessing import load_data, preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.header("📊 Model Evaluation")

# Load and preprocess
train, _, _ = load_data()
train = preprocess(train)

X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train['Survived']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)
st.metric("Training Accuracy", f"{acc*100:.2f}%")

# Classification Report
st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
