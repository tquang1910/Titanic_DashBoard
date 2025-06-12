# 🚢 Titanic Survival Dashboard

A fully interactive Streamlit dashboard to explore, visualize, predict and evaluate survival data from the famous Titanic dataset.  
This project was refactored into a clean **multi-page Streamlit app**, modularized for easy maintenance, readability and extension.

---

## 📊 Features

- ✅ Cleaned Titanic dataset with proper preprocessing
- ✅ Descriptive data overview
- ✅ Interactive visualizations with filters (Gender, Class, Embarkation Port, Age)
- ✅ Survival prediction using RandomForestClassifier
- ✅ Model evaluation with classification report and confusion matrix
- ✅ Fully modular code structure with separate `pages/` and `utils/`

---

## 🗂 Project Structure

```bash
titanic_dashboard_project/
│
├── app.py                  # Main entry point for Streamlit
│
├── pages/                  # Multi-page Streamlit modules
│   ├── 1_📊_Overview.py
│   ├── 2_📈_Visualizations.py
│   ├── 3_🔮_Prediction.py
│   └── 4_📊_Model Evaluation.py
│
├── utils/                  # Data preprocessing utilities
│   └── preprocessing.py
│
├── data/                   # Dataset folder
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
└── requirements.txt        # List of required Python packages

## ⚙️ Install & Run Locally

1️⃣ Clone repository

```bash
git clone https://github.com/tquang1910/Titanic_DashBoard.git
cd Titanic_DashBoard
```

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
streamlit run app.py
```
## 🔧 Deployment
- Streamlit Cloud
- Render
- Railway

# Make sure requirements.txt and data/ folder exist in repository.
