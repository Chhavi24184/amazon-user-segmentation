import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def generate_synthetic_data(n=200,seed=42):
     np.random.seed(seed)
     data=pd.DataFrame({
         'User_ID': range(1, n+1),
        'Age': np.random.randint(18, 70, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Annual_Income': np.random.randint(15000, 200000, n),
        'Purchase_Frequency': np.random.randint(1, 30, n),
        'Avg_Spending': np.random.randint(100, 25000, n),
        'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Books', 'Sports'], n),
        'Review_Score': np.random.randint(1, 6, n)
     })
     return data
def load_or_create_data(path):
    if os.path.exists(path):
        print(f"Loading data from {path}")
        return pd.read_csv(path)
    else:
        print(f"{path} not foud -geneating synthetic dataset.")
        df=generate_synthetic_data()
        df.to_csv(path,index=False)
        print(f"Synthetic data saved to {path}")    
        return df
    
def preprocess(df):
    df=df.copy()
    df=df.dropna().reset_index(drop=True)   
    le_gender=LabelEncoder()
    le_cat=LabelEncoder()
    df['Gender']=le_gender.fit_transform(df['Gender'])
    df['Product_Category']=le_cat.fit_transform(df['Product_Category']) 
    features=['Age','Gender','Annual_Income','Purchase_Frequency','Avg_Spending','Product_Category','Review_Score']
    X=df[features].astype(float)    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return df,X_scaled,features
        