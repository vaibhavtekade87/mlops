import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

# Login to Hugging Face
login(token=os.environ["HF_TOKEN"])

# Load dataset from Hugging Face
dataset = load_dataset("tekadevaibhav/tourism-package-prediction", data_files="data/tourism.csv")
df = pd.DataFrame(dataset['train'])

# Drop unnecessary columns
df = df.drop(columns=['CustomerID'])

# Handle missing values
df = df.fillna({
    'Age': df['Age'].median(),
    'TypeofContact': df['TypeofContact'].mode()[0],
    'DurationOfPitch': df['DurationOfPitch'].median(),
    'NumberOfFollowups': df['NumberOfFollowups'].median(),
    'PreferredPropertyStar': df['PreferredPropertyStar'].median(),
    'NumberOfTrips': df['NumberOfTrips'].median(),
    'NumberOfChildrenVisiting': df['NumberOfChildrenVisiting'].median(),
    'MonthlyIncome': df['MonthlyIncome'].median()
})

# Split data
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save locally
train_df = X_train.copy()
train_df['ProdTaken'] = y_train
test_df = X_test.copy()
test_df['ProdTaken'] = y_test

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Data preparation completed")
