import pandas as pd
import os
import kagglehub
import glob



def load_data(path_name="blastchar/telco-customer-churn"):
    path = kagglehub.dataset_download(path_name)
    csv=glob.glob(os.path.join(path,"*.csv"))
    df = pd.read_csv(csv[0])
    return df

    