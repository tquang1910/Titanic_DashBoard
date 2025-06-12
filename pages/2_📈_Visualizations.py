import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import load_data, preprocess

st.header("📈 Interactive Visualizations")

# Load and preprocess
train, _, _ = load_data()
train = preprocess(train)

# Mapping for filters
sex_map = {0: 'Male', 1: 'Female'}
pclass_map = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

# Inverse mapping for filter values
sex_map_inv = {v: k for k, v in sex_map.items()}
pclass_map_inv = {v: k for k, v in pclass_map.items()}
embarked_map_inv = {v: k for k, v in embarked_map.items()}

# Sidebar filters
st.sidebar.header("🔧 Filters")
sex_labels = [sex_map.get(x, x) for x in train['Sex'].unique()]
sex_filter = st.sidebar.multiselect("Select Gender", options=sex_labels, default=sex_labels)
sex_filter_values = [sex_map_inv[x] for x in sex_filter]

pclass_labels = [pclass_map.get(x, x) for x in train['Pclass'].unique()]
pclass_filter = st.sidebar.multiselect("Select Ticket Class", options=pclass_labels, default=pclass_labels)
pclass_filter_values = [pclass_map_inv[x] for x in pclass_filter]

embarked_labels = [embarked_map.get(x, x) for x in train['Embarked'].unique()]
embarked_filter = st.sidebar.multiselect("Select Port of Embarkation", options=embarked_labels, default=embarked_labels)
embarked_filter_values = [embarked_map_inv[x] if x in embarked_map_inv else x for x in embarked_filter]

age_range = st.sidebar.slider("Select Age Range", int(train['Age'].min()), int(train['Age'].max()), (0, 80))

# Filter the dataset
filtered_df = train[
    (train['Sex'].isin(sex_filter_values)) &
    (train['Pclass'].isin(pclass_filter_values)) &
    (train['Embarked'].isin(embarked_filter_values)) &
    (train['Age'] >= age_range[0]) &
    (train['Age'] <= age_range[1])
]

# Relabel for display in charts
filtered_df['Sex'] = filtered_df['Sex'].map(sex_map)
filtered_df['Pclass'] = filtered_df['Pclass'].map(pclass_map)
filtered_df['Embarked'] = filtered_df['Embarked'].map(embarked_map)
filtered_df['Survived'] = filtered_df['Survived'].map({0: 'Did not survive', 1: 'Survived'})

# Chart selection
chart_type = st.selectbox("Choose Chart Type", ["Survival Pie Chart", "Age Boxplot", "Age Histogram", "Gender Barplot"])

if chart_type == "Survival Pie Chart":
    survived_counts = filtered_df['Survived'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(survived_counts, labels=survived_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.set_title("Survival Distribution")
    ax.axis('equal')
    st.pyplot(fig)

elif chart_type == "Age Boxplot":
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='Sex', y='Age', hue='Survived', palette="Set2", ax=ax)
    ax.set_title("Age Distribution by Gender and Survival")
    st.pyplot(fig)

elif chart_type == "Age Histogram":
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='Age', hue='Survived', kde=True, multiple="stack", bins=30, palette="pastel", ax=ax)
    ax.set_title("Age Histogram by Survival")
    st.pyplot(fig)

elif chart_type == "Gender Barplot":
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Sex', hue='Survived', palette='Set1', ax=ax)
    ax.set_title("Survival Rate by Gender")
    st.pyplot(fig)
