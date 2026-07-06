# =========================
# IMPORT REQUIRED LIBRARIES
# =========================
import numpy as np
import pandas as pd
import pickle
import os

from flask import Flask, request, render_template


# =========================
# CREATE FLASK APPLICATION
# =========================
app = Flask(__name__)


# =========================
# LOAD TRAINED MODEL & SCALER
# =========================
model = pickle.load(open('rdf.pkl', 'rb'))
scale = pickle.load(open('scale1.pkl', 'rb'))


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('home.html')


# =========================
# INPUT PAGE
# =========================
@app.route('/predict')
def predict():
    return render_template('input.html')


# =========================
# PREDICTION PAGE
# =========================
@app.route('/submit', methods=['POST'])
def submit():

    # Retrieve values from HTML form
    input_feature = [int(x) for x in request.form.values()]

    # Convert input into NumPy array
    input_feature = [np.array(input_feature)]

    # Column names
    names = [
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area'
    ]

    # Create DataFrame
    data = pd.DataFrame(input_feature, columns=names)

    # Scale the input data
    data = scale.transform(data)

    # Predict loan approval
    prediction = model.predict(data)

    # Convert prediction into integer
    prediction = int(prediction[0])

    # Display result
    if prediction == 0:
        result = "Loan will NOT be Approved"
    else:
        result = "Loan will be Approved"

    # Render result page
    return render_template('output.html', result=result)


# =========================
# RUN APPLICATION
# =========================
if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    app.run(host='0.0.0.0', port=port, debug=True)
