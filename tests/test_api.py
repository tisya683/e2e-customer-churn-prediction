# test client alllows you to test without uvicorn ui

from fastapi.testclient import TestClient
from app.main import app

test_version=TestClient(app)

example_customer={
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 79.85,
    "total_charges": 940.2,
}

#assert is a data validation statement. it checks if it works,if it doesnt work the test fails and pytest shows the actual val

def test_health_check():
    resp=test_version.get("/health")

    assert resp.status_code==200
    assert resp.json()=={"status":"ok"} #gets the response

def test_predict_churn():
    resp=test_version.post("/predict",json=example_customer)

    assert resp.status_code==200

    data=resp.json() #gets the response

    #check if will churn and churn probability exists 
    assert "will_churn" in data
    assert "churn_probability" in data
    #checks value of will churn and churn proabability
    assert isinstance(data["will_churn"],bool)
    assert 0<=data["churn_probability"]<=1

def test_int_field():
    invalid=example_customer.copy()
    invalid["tenure"]=12.3

    response=test_version.post("/predict",json=invalid)

    assert response.status_code==422 # shows that request body failed validation

def test_float_field():
    invalid=example_customer.copy()
    invalid["monthly_charges"]="randnadasdss"
    
    response=test_version.post("/predict",json=invalid)
    
    assert response.status_code==422 

def test_missing_value():
    invalid=example_customer.copy()
    del invalid["contract"]
    
    response=test_version.post("/predict",json=invalid)
    
    assert response.status_code==422 
