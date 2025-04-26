# Step 1: Use an official Python image as the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file (assumes requirements.txt is in the repo)
COPY requirements.txt .

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Expose the port your application will run on (e.g., 5000 for Flask)
EXPOSE 5000

# Step 7: Set environment variables if needed
# ENV VARIABLE_NAME=value

# Step 8: Define the command to run your application
# This assumes you are running a Flask or FastAPI app (or similar)
CMD ["python", "app.py"]  # Adjust the entry point as needed
