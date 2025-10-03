# Dockerfile

# Use an official Python runtime as a parent image.
# 'slim' is a smaller version, great for production.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgdal-dev g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container first
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (main.py) and data (koppen.tif) into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app using Gunicorn
# Gunicorn is a robust production server that manages Uvicorn workers.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]