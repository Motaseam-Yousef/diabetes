# Diabetes Prediction App 

## Overview
This **Diabetes Prediction App** is built using **Streamlit** to predict whether a person is likely to develop diabetes based on their health parameters. The app takes various inputs, such as pregnancies, glucose levels, blood pressure, BMI, and more, and uses a pre-trained machine learning model to provide a prediction.

## Features
- **User-friendly interface**: Input fields are neatly organized into two columns for better readability.
- **Interactive prediction**: Users can input values and receive a prediction by clicking the "Predict" button.
- **Custom styling**: The app uses custom CSS for a clean and modern look.
- **Diabetes Prediction Model**: The app leverages a pre-trained ensemble model (loaded from a file) to make predictions based on input parameters.

## Setup Instructions

### Prerequisites
Before running the application, ensure you have the following:
- Python 3.x installed.
- Necessary dependencies installed from `requirements.txt`.

### Folder Structure
Make sure your project has the following structure:

```
project_root/
│
├── model/
│   ├── scaler.pkl
│   └── ensemble_model.pkl
│
├── app.py
├── requirements.txt
```

- `scaler.pkl`: The scaler used for normalizing input data.
- `ensemble_model.pkl`: The trained ensemble model for diabetes prediction.
- `app.py`: The main script that runs the Streamlit app.
- `requirements.txt`: The file containing the necessary Python libraries.

### Installing Dependencies
To install the required dependencies, run the following command in your project directory:

```bash
pip install -r requirements.txt
```

### Running the App
Once dependencies are installed, you can run the app by executing the following command in the terminal:

```bash
streamlit run app.py
```

The app will open in your browser, and you can input health parameters to predict the likelihood of diabetes.

## How the App Works

1. **Loading Models**: The app loads the pre-trained scaler and ensemble model from the `model/` directory using Joblib.
2. **User Input**: The app collects the following user inputs:
    - Pregnancies
    - Glucose
    - Blood Pressure
    - BMI
    - Skin Thickness
    - Insulin
    - Diabetes Pedigree Function
    - Age
3. **Scaling Input Data**: The inputs are scaled using the loaded `scaler.pkl` model.
4. **Prediction**: The scaled data is passed to the loaded `ensemble_model.pkl`, which predicts whether the person is likely to have diabetes (binary output: 0 or 1).
5. **Displaying Results**: Based on the prediction, the app displays:
    - An error message if the prediction is positive (likely to have diabetes).
    - A success message if the prediction is negative (unlikely to have diabetes).