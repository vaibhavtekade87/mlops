# Tourism Package Prediction - MLOps Pipeline

An end-to-end MLOps pipeline for predicting customer purchases of Wellness Tourism Packages.

## Objective
Predict whether a customer will purchase the Wellness Tourism Package using machine learning, with automated CI/CD deployment.

## Architecture
Data (HuggingFace) -> Training -> Model Registry (HuggingFace) -> Streamlit App (HuggingFace Spaces)
                          |
                    GitHub Actions (CI/CD)

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 93.3% |
| Precision | 91.9% |
| Recall | 71.7% |
| F1-Score | 80.5% |

## Links
- Live App: https://huggingface.co/spaces/tekadevaibhav/tourism-package-app
- Dataset: https://huggingface.co/datasets/tekadevaibhav/tourism-package-prediction
- Model: https://huggingface.co/tekadevaibhav/tourism-package-model

## Tech Stack
Python, XGBoost, Scikit-learn, Streamlit, Docker, GitHub Actions, HuggingFace

## Project Structure
.github/workflows/pipeline.yml  - CI/CD Pipeline
src/                            - ML Scripts
deployment/                     - Docker and Streamlit
data/                           - Dataset
requirements.txt                - Dependencies

## Quick Start
git clone https://github.com/vaibhavtekade87/mlops.git
pip install -r requirements.txt

