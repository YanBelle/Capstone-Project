# Update Jupyter Dockerfile
FROM jupyter/scipy-notebook:latest

USER root
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

USER $NB_UID
RUN pip install --no-cache-dir torch==2.0.1 transformers==4.31.0 psycopg2-binary==2.9.7 sqlalchemy==2.0.19 redis==4.6.0 plotly==5.15.0 joblib==1.3.1 scikit-learn==1.3.0

# Don't copy notebooks in Dockerfile, use volume mount instead
WORKDIR /home/jovyan