FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install only the system packages required by
# Pillow / TensorFlow and basic runtime functionality.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching.
COPY backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the Django backend.
COPY backend/ /app/backend/

# Copy the production ML artifacts required by model_manager.py.
COPY artifacts/ /app/artifacts/

WORKDIR /app/backend

# Django runs on port 8000 inside the container.
EXPOSE 8000

# Apply database migrations when the container starts,
# then launch the Django application.
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]