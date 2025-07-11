FROM python:3.11-slim

WORKDIR /app

# 1) Install system build deps needed for numpy/scipy/statsmodels/prophet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      libatlas-base-dev \
      libblas-dev \
      liblapack-dev \
      python3-dev \
      cmake \
      libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your code & expose port
COPY . .
EXPOSE 8501

# 4) Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
