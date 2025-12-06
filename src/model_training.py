import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from huggingface_hub import login, HfApi, create_repo

# Login to Hugging Face
login(token=os.environ["HF_TOKEN"])

# Load data
train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']

# Define columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [4, 6]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best model
os.makedirs("model", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "model/best_model.pkl")

# Upload to Hugging Face
api = HfApi()
create_repo(repo_id="tekadevaibhav/tourism-package-model", repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="model/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id="tekadevaibhav/tourism-package-model",
    repo_type="model"
)

print("Model training completed")
print("Best Parameters:", grid_search.best_params_)
