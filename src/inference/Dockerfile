# ─────────────────────────────────────────
# Dockerfile — inference step
# Generates predictions using trained model
# ─────────────────────────────────────────

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy step source code
COPY inference.py .
COPY utils/ utils/
COPY __main__.py .

# Entry point
ENTRYPOINT ["python", "__main__.py"]