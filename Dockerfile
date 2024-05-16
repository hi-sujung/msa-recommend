# Create a new build stage from a base image
FROM python:3.12-slim

# Change working directory
WORKDIR /app

# Copy necessary files and directories
COPY requirements.txt .
COPY app.py .
COPY config.py .  

# Execute build commands
RUN pip install --no-cache-dir -r requirements.txt

# Describe which ports your application is listening on
EXPOSE 5000

# Specify default executable
ENTRYPOINT [ "flask", "run", "--host=0.0.0.0" ]
