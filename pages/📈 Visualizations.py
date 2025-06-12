
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import load_data, preprocess

st.header("📈 Interactive Visualizations (Enhanced Layout)")

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

# Mapping lại đúng với số hóa sau encode
sex_map = {0: 'Male', 1: 'Female'}
pclass_map = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
embarked_map = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}
survived_map = {0: 'Did not survive', 1: 'Survived'}

# Apply mapping
filtered_df['Sex'] = filtered_df['Sex'].map(sex_map)
filtered_df['Pclass'] = filtered_df['Pclass'].map(pclass_map)
filtered_df['Embarked'] = filtered_df['Embarked'].map(embarked_map)
filtered_df['Survived'] = filtered_df['Survived'].map(survived_map)

# --- Multiple Charts Layout ---
st.subheader("📊 Summary Visualizations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧍 Survival by Gender**")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Sex', hue='Survived', palette="Set2", ax=ax1)
    ax1.set_title("Survival by Gender")
    st.pyplot(fig1)

with col2:
    st.markdown("**🎟️ Survival by Class**")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered_df, x='Pclass', hue='Survived', palette="Set2", ax=ax2)
    ax2.set_title("Survival by Passenger Class")
    st.pyplot(fig2)

with col3:
    st.markdown("**⚓ Port of Embarkation**")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=filtered_df, x='Embarked', hue='Survived', palette="Set2", ax=ax3)
    ax3.set_title("Survival by Embarkation Port")
    st.pyplot(fig3)

st.markdown("---")

col4, col5 = st.columns(2)
with col4:
    st.markdown("**📊 Age Distribution**")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=filtered_df, x='Age', hue='Survived', bins=30, kde=True, palette="Set2", ax=ax4)
    ax4.set_title("Age Histogram by Survival")
    st.pyplot(fig4)

with col5:
    st.markdown("**📈 Survival Proportion**")
    survived_counts = filtered_df['Survived'].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.pie(survived_counts, labels=survived_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    ax5.set_title("Survival Distribution")
    ax5.axis('equal')
    st.pyplot(fig5)
