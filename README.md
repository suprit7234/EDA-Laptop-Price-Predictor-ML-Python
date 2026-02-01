# 💻 Laptop Price Predictor (Machine Learning + Streamlit)

A complete end-to-end Machine Learning project that predicts the price of a laptop based on its configuration such as brand, RAM, storage, CPU, GPU, screen size, resolution, and display features. The model is deployed as an interactive web application using Streamlit.

---https://github.com/suprit7234

## 🚀 Project Overview

This project covers the full ML workflow:
- Data preprocessing and feature engineering  
- Model training using a Scikit-learn pipeline  
- Model serialization using Pickle  
- Web app deployment using Streamlit  

Users can select laptop specifications from a UI and get an estimated price instantly.

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Web Framework:** Streamlit  
- **Model Serialization:** Pickle  
- **Notebook:** Jupyter  

---

## ✨ Features

- Interactive UI for selecting laptop configuration  
- Handles categorical and numerical features using ML pipeline  
- Calculates PPI (Pixels Per Inch) from screen resolution and size  
- Predicts laptop price using trained regression model  
- Clean and beginner-friendly code structure  

---

## 📁 Project Structure

Laptop-Pred-ML/
│
├── Lap-app.py # Streamlit web application
├── Laptop-Pred.ipynb # Model training & experimentation notebook
├── laptop-data.csv # Dataset used for training
├── pipe.pkl # Trained ML pipeline (model + preprocessing)
├── df.pkl # Dataframe for dropdown values in UI
└── README.md # Project documentation

## ⚙️ How to Run Locally

1️⃣ Clone the repository:
```bash
git clone https://github.com/suprit7234/laptop-price-predictor.git
cd laptop-price-predictor

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run Lap-app.py
```
Model Details:
Type: Regression model with preprocessing pipeline (ColumnTransformer)
Features: Company, TypeName, RAM, Weight, Touchscreen, IPS, Inches, PPI, CPU, HDD, SSD, GPU, OS
Target: Laptop price (log-transformed during training)

Author
Suprit
GitHub: https://github.com/suprit7234
LinkedIn: www.linkedin.com/in/suprit-tapase-637541294
Email: suprittapase3@gmail.com
