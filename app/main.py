from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn


#connect to mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#load best model
MODEL_URI = "models:/churn_prediction_model/1"
model = mlflow.sklearn.load_model(MODEL_URI)


#create. fastapi app instance 
app=FastAPI(
    title="telco customer churn prediction api",#displayed name in the docs
    version="1.0.0"
)

# input features for model
# BaseModel automatically checks and validates the incoming data against the fields and types you define
class CustomerInput(BaseModel):
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


#check if api is running
@app.get("/health")
def health_check():
    return{"status":"ok"}

#predict for only one custoemr
#run the model in the app
#model_dump converts the input pydantic data into python dict
@app.post("/predict")
def predict_churn(customer:CustomerInput):
    customer_df=pd.DataFrame([customer.model_dump()])

    prediction=int(model.predict(customer_df)[0])# will return an arrahy. need to include 0 to get whats inside array
    churn_probability=float(model.predict_proba(customer_df)[0,1])#first customer, probability fo churning

    return{
        "will_churn": bool(prediction),
        "churn_probability": round(churn_probability,2)#round to 2.d.p
    }


