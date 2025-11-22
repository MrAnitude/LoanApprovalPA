import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Starter Data
loanApprovalData = pd.read_csv("Loan_approval_data_2025.csv")
print(loanApprovalData.head())
print(loanApprovalData.columns)

# Statistical data Table (count, mean, std, etc.)
print(loanApprovalData.describe())
print(loanApprovalData.info())

# Dimensions of Data
print(loanApprovalData.shape)

# ------ Data Clean Up -------

loanApprovalData = loanApprovalData.drop(columns=["customer_id"])
print(loanApprovalData)

'''
One Hot Encoding:
a data preprocessing technique that converts categorical data into a numerical format for machine learning algorithms. 
It works by creating a new binary column for each unique category, where a '1' indicates the presence of that category in a row and '0' indicates its absence
turns a single column into mulitple columns of unique catagories from the original column and sets the correct new column to 1 and the rest of them to 0
'''
from sklearn.preprocessing import OneHotEncoder

'''
Column Transformer:
Performs different transformation on different column like Imputation on numerical data, One Hot Encoding on Categorical data and creates a new array all at once
'''
from sklearn.compose import ColumnTransformer

'''
Standard Scaler(Data Scaling):
To keep certain columns larger ranges from over-powering other columns 
Standardizing all the columns ranges to a range of 0-1, -3 - 3, -1-1, etc.
allows for all features to contribute fairly to predictions
'''
from sklearn.preprocessing import StandardScaler


transformer  = ColumnTransformer([("one_hot_encoder", OneHotEncoder(sparse_output= False), ["occupation_status", "product_type", "loan_intent"])], remainder= "passthrough")

print(loanApprovalData["occupation_status"].unique())
print(loanApprovalData["product_type"].unique())
print(loanApprovalData["loan_intent"].unique())

# transformer.fit_transform moves all the transformed data to the front of the array (1 column turns into 3, 3-1 = +2 columns total)
tladf = transformer.fit_transform(loanApprovalData)
print(tladf[0])
print(tladf.shape)