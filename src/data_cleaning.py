
import pandas as pd
import numpy as np
import re



def clean_data(df):
    df = df.copy()

    # convert names such as customerID and streamingTV to snake_case.
    df.columns = [
        re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column).lower()
        for column in df.columns
    ]

    df["total_charges"] = pd.to_numeric(
        df["total_charges"],
        errors="coerce",
    )

    df = df.drop_duplicates()

    # keep senior_citizen as a categorical Yes/No field.
    df["senior_citizen"] = df["senior_citizen"].map({
        1: "Yes",
        0: "No",
    })

    # remove the 11 records with missing total_charges only.
    df = df.dropna(subset=["total_charges"])

    # convert the prediction target to numeric.
    df["churn"] = df["churn"].map({
        "Yes": 1,
        "No": 0,
    })

    return df


