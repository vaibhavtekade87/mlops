import os
from huggingface_hub import login, HfApi, create_repo

# Login to Hugging Face
login(token=os.environ["HF_TOKEN"])

api = HfApi()

# Create Space
create_repo(
    repo_id="tekadevaibhav/tourism-package-app",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

# Upload files
api.upload_file(path_or_fileobj="deployment/Dockerfile", path_in_repo="Dockerfile", repo_id="tekadevaibhav/tourism-package-app", repo_type="space")
api.upload_file(path_or_fileobj="deployment/app.py", path_in_repo="app.py", repo_id="tekadevaibhav/tourism-package-app", repo_type="space")
api.upload_file(path_or_fileobj="deployment/requirements.txt", path_in_repo="requirements.txt", repo_id="tekadevaibhav/tourism-package-app", repo_type="space")

print("Deployment to Hugging Face Space completed")
