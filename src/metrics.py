from sklearn.metrics import (precision_score,recall_score,f1_score,average_precision_score,precision_recall_curve)
import matplotlib.pyplot as plt


#metrics
def compute_metrics(y_actual,y_pred,y_scores,plot=False,label="Model"):
       ap=average_precision_score(y_actual, y_scores)

       if plot:
              precision,recall, _= precision_recall_curve(y_actual,y_scores)
              baseline=y_actual.mean() #positive class rate

              plt.figure(figsize=(6, 5))
              plt.plot(recall,precision,label=f"{label}(AP={ap:.3f})")
              plt.axhline(baseline,linestyle="--",color="gray",label=f"Baseline({baseline:.3f})")
              plt.xlabel("Recall")
              plt.ylabel("Precision")
              plt.title("Precision-Recall Curve")
              plt.legend(loc="best")
              plt.grid(alpha=0.3)
              plt.show()
      
       return {
        "Precision": precision_score(y_actual, y_pred, zero_division=0),
        "Recall": recall_score(y_actual, y_pred, zero_division=0),
        "F1": f1_score(y_actual, y_pred, zero_division=0),
        "PR-AUC": ap
    }