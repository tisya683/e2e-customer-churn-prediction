
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_x_y(df):
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    return X, y


def create_preprocessor(X):
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def split_data(X, y):
   X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
   return X_train,X_test,y_train,y_test