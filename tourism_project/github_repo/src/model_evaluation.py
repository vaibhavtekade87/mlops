import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from huggingface_hub import login

# Login to Hugging Face
login(token=os.environ["HF_TOKEN"])

# Load data and model
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=['ProdTaken'])
y_test = test_df['ProdTaken']

model = joblib.load("model/best_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Model Evaluation Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
