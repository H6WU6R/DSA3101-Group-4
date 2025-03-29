# Use an official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies from requirements.txt (if available)
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port (if needed, e.g., for a web app)
EXPOSE 5000

# Set the default command to run your application
CMD ["python", "main.py"]

