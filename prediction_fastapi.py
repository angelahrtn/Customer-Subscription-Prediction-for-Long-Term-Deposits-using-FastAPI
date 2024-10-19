# Import libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialize FastAPI application
app = FastAPI()

# Load model, scaler, and encoders
try:
    with open('one_hot_encoding.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('label_encoding.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading resources: {e}")
    raise HTTPException(status_code=500, detail="Failed to load necessary resources")

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Endpoint for making predictions
@app.post("/predict/")
async def make_prediction(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Apply one-hot encoding and label encoding
        for column, encoder in encoders.items():
            if column in input_data:
                input_data = pd.concat([input_data.drop(column, axis=1),
                                        pd.DataFrame(encoder.transform(input_data[[column]]),
                                        columns=[f"{column}_{lvl}" for lvl in encoder.classes_])], axis=1)

        for label_col, mapping in label_encoders.items():
            if label_col in input_data:
                input_data[label_col] = input_data[label_col].map(mapping)

        # Scale numeric features
        numeric_features = ['age', 'campaign', 'previous', 'duration', 'pdays']
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

        # 'expected_features' is a list of the model's features 
        expected_features = ['age', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'job_admin.', 'job_blue-collar', 'job_entrepreneur',
       'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
       'job_services', 'job_student', 'job_technician', 'job_unemployed',
       'job_unknown', 'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school', 'education_illiterate',
       'education_professional.course', 'education_university.degree',
       'education_unknown', 'default_no', 'default_unknown', 'housing_no',
       'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
       'poutcome_failure', 'poutcome_nonexistent', 'poutcome_success',
       'contact_cellular', 'contact_telephone']  # all features

        # Reindex the DataFrame to match training features
        # input_data has the same feature order as the training data
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction result
        result = 'yes' if prediction[0] == 1 else 'no'
        return {"prediction": result}

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))