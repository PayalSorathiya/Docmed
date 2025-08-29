# Disease Prediction System

A machine learning-based medical diagnosis system that predicts diseases based on symptoms and provides comprehensive health recommendations including medications, diet plans, precautions, and doctor referrals.

## Overview

This project uses ensemble machine learning techniques to predict diseases from patient symptoms. The system combines Gaussian Naive Bayes and Logistic Regression algorithms using a Voting Classifier to achieve high accuracy in disease prediction.

## Features

- **Disease Prediction**: Predicts from 41 different diseases based on input symptoms
- **Comprehensive Health Information**: Provides detailed information including:
  - Disease descriptions
  - Precautionary measures
  - Related symptoms
  - Recommended medications
  - Diet suggestions
  - Workout/lifestyle recommendations
  - Doctor referrals with specialization

## Dataset

The system uses multiple CSV files containing:
- **Training.csv**: Main dataset with 132 symptoms and 41 diseases (304 records)
- **symtoms_df.csv**: Disease-symptom mappings
- **precautions_df.csv**: Precautionary measures for each disease
- **description.csv**: Disease descriptions
- **medications.csv**: Recommended medications
- **diets.csv**: Dietary recommendations
- **workout_df.csv**: Lifestyle and workout suggestions
- **Doctor.csv**: Doctor information and specializations

## Model Performance

- **Accuracy**: 100%
- **RMSE**: 0.0
- **Precision**: 1.00
- **F1 Score**: 1.00
- **Recall**: 1.00

## Algorithm

The system uses a **Voting Classifier** combining:
- **Gaussian Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Logistic Regression**: Linear model for binary classification

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disease-prediction-system.git
cd disease-prediction-system

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn pickle
```

## Usage

### 1. Training the Model

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle

# Load and prepare data
df = pd.read_csv("Training.csv")
df = df.drop_duplicates()

# Split features and target
x = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train the model
gnb = GaussianNB()
lr = LogisticRegression(random_state=42)
voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', lr)], voting='soft')
voting_clf.fit(x_train, y_train)

# Save the model
pickle.dump(voting_clf, open('model.pkl', 'wb'))
```

### 2. Making Predictions

```python
# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Input symptoms (comma-separated)
symptoms = input("Enter your symptoms: ")
patient_symptoms = [s.strip().lower() for s in symptoms.split(',')]

# Get prediction
predicted_disease = get_prediction(patient_symptoms)
print(f"Predicted Disease: {predicted_disease}")
```

### 3. Example Usage

```
Enter your symptoms: muscle_wasting,loss_of_balance,visual_disturbances

Predicted Disease: AIDS

Description:
AIDS (Acquired Immunodeficiency Syndrome) is a disease caused by HIV that weakens the immune system.

Precautions:
1: avoid open cuts
2: wear ppe if possible
3: consult doctor
4: follow up

Related symptoms:
1: muscle_wasting
2: patches_in_throat
3: high_fever
4: extra_marital_contacts

Medications:
['Antiretroviral drugs', 'Protease inhibitors', 'Integrase inhibitors', 'Entry inhibitors', 'Fusion inhibitors']

Diets:
['Balanced Diet', 'Protein-rich foods', 'Fruits and vegetables', 'Whole grains', 'Healthy fats']

Doctor Referral:
Specialization: Immunologist
Gender: F
DoctorID: D44
```

## Available Symptoms

The system recognizes 132 different symptoms including:
- itching, skin_rash, nodal_skin_eruptions
- continuous_sneezing, shivering, chills
- joint_pain, stomach_pain, acidity
- muscle_wasting, vomiting, burning_micturition
- fatigue, weight_gain, anxiety
- And many more...

## Supported Diseases

The system can predict 41 different diseases:
- Fungal infection, Allergy, GERD
- Chronic cholestasis, Drug Reaction
- Peptic ulcer disease, AIDS, Diabetes
- Gastroenteritis, Bronchial Asthma
- Hypertension, Migraine, Cervical spondylosis
- And many more...

## Project Structure

```
disease-prediction-system/
│
├── DocMed.ipynb              # Main Jupyter notebook
├── model.pkl                 # Trained model file
├── Training.csv              # Main training dataset
├── symtoms_df.csv           # Symptoms mapping
├── precautions_df.csv       # Precautions data
├── description.csv          # Disease descriptions
├── medications.csv          # Medication recommendations
├── diets.csv               # Diet recommendations
├── workout_df.csv          # Workout suggestions
├── Doctor.csv              # Doctor information
└── README.md               # This file
```

## Key Functions

### `get_prediction(symptoms)`
- Takes a list of symptoms as input
- Returns predicted disease name
- Uses symptom-to-vector encoding

### `helper(disease)`
- Takes disease name as input
- Returns comprehensive information:
  - Description, precautions, medications
  - Diet recommendations, workout plans
  - Doctor referral information

## Technical Details

- **Data Preprocessing**: Label encoding for disease names, binary encoding for symptoms
- **Train-Test Split**: 70:30 ratio
- **Feature Vector**: 132-dimensional binary vector representing symptoms
- **Model Persistence**: Uses pickle for model serialization

## Limitations

- Predictions are based on training data and should not replace professional medical advice
- Always consult healthcare professionals for accurate diagnosis
- The system is designed for educational and reference purposes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request



## Disclaimer

This system is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.



**Note**: Ensure all CSV files are in the same directory as the notebook for proper functionality.
