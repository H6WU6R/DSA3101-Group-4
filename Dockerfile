# Use an official Python image as the base
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install GPG for decryption commands
RUN apt-get update && apt-get install -y gnupg

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first for caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . /app

# Expose a port if needed (for example, if your app runs a server)
# EXPOSE 5000

# Final command: runs main.py from the src folder
CMD ["python", "src/main.py"]
