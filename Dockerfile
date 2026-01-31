FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (make, gcc for some python libs if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching (optimization attempt)
COPY pyproject.toml .
COPY Makefile .
COPY Address_Points_*.csv .

# Copy Source Code (Required for pip install .)
COPY src/ src/
# Copy Data (Although we stick volume, good to have default)
COPY data/ data/

# Install dependencies and the project itself
RUN pip install --upgrade pip && \
    pip install .

# Expose Static Site Port
EXPOSE 8000

# Default Command: Check for DB, ensure export is fresh, and serve static site
CMD ["sh", "-c", "python -m summit_housing.ingestion && make export && make serve-static"]
