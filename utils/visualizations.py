import seaborn as sns
import matplotlib.pyplot as plt

def survival_by_sex(df):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Sex', hue='Survived', ax=ax)
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_title("Survival Rate by Gender")
    return fig

def survival_by_class(df):
    fig, ax = plt.subplots()
    sns.barplot(x='Pclass', y='Survived', data=df, ax=ax)
    ax.set_title("Survival Rate by Passenger Class")
    return fig

def age_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution of Passengers")
    return fig
