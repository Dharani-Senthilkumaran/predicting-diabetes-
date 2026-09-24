# 🩺 Diabetes Prediction Using Random Forest

A machine learning web application that predicts whether a person is likely to have diabetes based on medical and health-related input features.

The project uses a **Random Forest Classifier** for prediction and **Flask** to provide a simple web interface.

## 📌 Project Overview

Diabetes is a common health condition that can be influenced by factors such as glucose level, blood pressure, BMI, age, and other health-related parameters.

This project uses machine learning to analyze these features and predict the possibility of diabetes.

The trained Random Forest model is saved as a `.pkl` file and loaded by the Flask application to make predictions without retraining the model every time the application starts.

## 🚀 Features

* Diabetes prediction using Random Forest
* Simple and user-friendly web interface
* Flask-based backend
* Pre-trained machine learning model
* Fast prediction without retraining
* Input validation and prediction result display
* HTML/CSS frontend

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **HTML**
* **CSS**
* **Machine Learning – Random Forest**

## 📊 Dataset

The project uses the **Pima Indians Diabetes Database**.

The dataset contains the following features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age
* Outcome

`Outcome` is the target variable:

* `0` → No Diabetes
* `1` → Diabetes

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier**.

Random Forest combines multiple decision trees to make a final prediction. It can capture non-linear relationships between medical features and the target variable and is suitable for classification problems.

The trained model is stored in:

```text
models/random_forest_model.pkl
```

## 📁 Project Structure

```text
predictive_diabetes/
│
├── app.py
│
├── dataset/
│   └── diabetes.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── prediction_form.html
│   └── result.html
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Divya22022004/predictive_diabetes.git
```

### 2. Open the project folder

```bash
cd predictive_diabetes
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open the URL displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

Enter the required health information and submit the form to receive the prediction.

## 🔄 How It Works

```text
User Input
    ↓
Flask Web Application
    ↓
Pre-trained Random Forest Model
    ↓
Feature Processing
    ↓
Diabetes Prediction
    ↓
Result Displayed to User
```

## 🎯 Project Objective

The main objective of this project is to demonstrate how machine learning can be integrated with a web application to perform diabetes prediction using patient-related input features.

## 🔮 Future Enhancements

* Deploy the application to a cloud platform
* Improve model performance using additional datasets
* Add model explainability and feature importance visualization
* Improve UI/UX
* Add secure user authentication
* Provide more detailed prediction insights

## 👩‍💻 Project Team

**Divya R**
**Dharani**
**Durga**

## ⚠️ Disclaimer

This application is developed for educational and demonstration purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

