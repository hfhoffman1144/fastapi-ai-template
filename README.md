# fastapi-ai-template
Control your own destiny and cut costs - A FastAPI template for deploying open source and custom AI models at scale.

# TODO
- Running locally

docker build --no-cache -t fastapi-ai-template:latest .

docker run -p 8000:8000 -v /Users/HarrisonHoffman/Desktop/DataScience/fastapi-ai-template/medium-demo-proj-c37770f6fe56.json:/app/gcp_credentials.json --env-file .env -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_credentials.json fastapi-ai-template:latest


