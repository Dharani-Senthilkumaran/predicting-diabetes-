from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the models
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/neural_network_model.pkl', 'rb') as f:
        nn_model = pickle.load(f)
    return rf_model, lr_model, nn_model

# Load the dataset and train models
def train_models():
    df = pd.read_csv('dataset/diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    
    nn_model = MLPClassifier()
    nn_model.fit(X_train, y_train)
    nn_accuracy = accuracy_score(y_test, nn_model.predict(X_test))
    
    # Save models
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/neural_network_model.pkl', 'wb') as f:
        pickle.dump(nn_model, f)
    
    return {
        'Random Forest': rf_accuracy,
        'Logistic Regression': lr_accuracy,
        'Neural Network': nn_accuracy
    }

# Define routes
@app.route('/')
def index():
    accuracies = train_models()
    return render_template('index.html', accuracies=accuracies)

@app.route('/prediction_form')
def prediction_form():
    return render_template('prediction_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    rf_model, lr_model, nn_model = load_models()
    
    try:
        features = [
            int(request.form['Pregnancies']),
            int(request.form['Glucose']),
            int(request.form['BloodPressure']),
            int(request.form['SkinThickness']),
            int(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            int(request.form['Age'])
        ]
    except KeyError as e:
        return render_template('result.html', error_message=f"Missing input field: {e}")
    except ValueError as e:
        return render_template('result.html', error_message=f"Invalid input value: {e}")

    features = np.array(features).reshape(1, -1)
    
    predictions = {
        'Random Forest': rf_model.predict(features)[0],
        'Logistic Regression': lr_model.predict(features)[0],
        'Neural Network': nn_model.predict(features)[0]
    }
    
    # Combine predictions: Majority vote
    result = np.mean(list(predictions.values())) > 0.5
    final_prediction = 'Diabetic' if result else 'Non-Diabetic'
    
    return render_template('result.html', prediction=final_prediction)

if __name__ == '__main__':
    app.run(debug=True)
