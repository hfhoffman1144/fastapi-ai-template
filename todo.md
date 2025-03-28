-  Rate limiting
- [X] Get transformers running 
- authentication
    - [X] Set API token expiration
    - [X] Create and test endpoint with authentication
- inference end points
    - [X] Streaming endpoint
    - [X] Determine which model to try and get code running in collab
    - [X] Create inference endpoint
    - [X] Make endpoints open AI compatible
    - [] Add db table to track requests and generate IDs
    - [] Handle different message types
- pre-commit
- CI/CD
    - [] Build image locally (need nvidia base image)
    - [] Get cloud build pipeline implemented
- Cloud run
    - Deploy and test
    - Use secrets instead of env variables


git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt
pip install -e . 

[project]
name = "fastapi-ai-template"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "accelerate>=1.3.0",
    "bcrypt>=4.2.1",
    "fastapi>=0.115.8",
    "google-cloud-bigquery-storage>=2.28.0",
    "google-cloud-bigquery>=3.29.0",
    "google-cloud-discoveryengine>=0.13.6",
    "httpx>=0.28.1",
    "ipykernel>=6.29.5",
    "openai>=1.65.5",
    "passlib>=1.7.4",
    "pre-commit>=4.1.0",
    "pyarrow>=19.0.0",
    "pydantic-settings>=2.7.1",
    "pyjwt>=2.10.1",
    "python-dotenv>=1.0.1",
    "python-multipart>=0.0.20",
    "safetensors>=0.5.2",
    "sqlalchemy-bigquery>=1.12.1",
    "sqlmodel>=0.0.22",
    "transformers>=4.48.2",
    "uvicorn>=0.34.0",
]


export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"