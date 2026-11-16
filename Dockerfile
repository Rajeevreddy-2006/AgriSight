FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src \
    AGRISIGHT_CONFIG=/app/configs/default.yaml

EXPOSE 8000 8501
# API by default. For the UI: docker run ... streamlit run app/streamlit_app.py --server.address 0.0.0.0
CMD ["uvicorn", "agrisight.api:app", "--host", "0.0.0.0", "--port", "8000"]
