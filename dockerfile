# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

 
CMD ["python", "main_shift_cycle.py"]

