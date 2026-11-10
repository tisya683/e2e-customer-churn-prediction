import mlflow
import mlflow.sklearn
from pathlib import Path

def main():
    #mlflow ui
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    #searches across all models in mlflow for the one named churn_prediciton_model which is the best lr model
    best_model=mlflow.sklearn.load_model("models:/churn_prediction_model/1")

    #create models folder for output
    #create parent directory if it doesnt exist
    output_path=Path("models/best_churn_model")
    output_path.parent.mkdir(parents=True,exist_ok=True)


    #save model locally
    mlflow.sklearn.save_model(best_model,str(output_path))
    print(f"model exported to:{output_path}")

if __name__ == "__main__":
    main()






