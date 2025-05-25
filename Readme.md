# 🍷 Wine Quality Prediction using Machine Learning

This project was created as a **personal milestone** to test and apply the machine learning knowledge I’ve gained so far. It predicts wine quality using chemical attributes and features a user-friendly interface built with **Streamlit**. This hands-on project helped me implement real-world ML techniques, understand model evaluation, and explore data preprocessing methods like **SMOTE** for class imbalance.

---

## 🎯 Purpose

To validate and apply the ML skills I’ve learned through self-study by building an end-to-end mini project:
- From **data preprocessing** to **model training**
- From **class imbalance handling** to **visual feature interpretation**
- All wrapped into an **interactive web app**

---

## 🧠 Technologies Used

- **Python 3**
- **Pandas**, **NumPy** – Data handling
- **Matplotlib**, **Seaborn** – Visualization
- **XGBoost** – Classification model
- **SMOTE** – Handling class imbalance
- **Scikit-learn** – ML evaluation
- **Streamlit** – Web-based UI

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app script |
| `wine-quality-final.xlsx` | Dataset used for training |
| `README.md` | This documentation |

---

## ⚙️ How it Works

1. **Loads data** from Excel
2. **Preprocesses it** – including column cleanup and conversion
3. **Binarizes target** – Good (≥7), Bad (<7)
4. **Handles imbalance** – using SMOTE
5. **Trains a model** – with XGBoost
6. **Builds a Streamlit app** – for user interaction & predictions

---

## 📊 Model Performance

- **Accuracy**: ~88–90% (varies by split)
- **ROC-AUC**: ~0.91
- **Feature Importance**: Visualized dynamically in the app

---

## 🚀 How to Run

> 📌 Make sure you update the file path to the dataset in `app.py`  
> Tip: Move the Excel file to the same directory as `app.py` and change the path to just the filename.

```bash
# Clone the repo
git clone https://github.com/<your-username>/wine-quality-prediction.git
cd wine-quality-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
