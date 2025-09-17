FROM python:3.11-slim

# Install ocrmypdf and its dependencies.
RUN apt-get update && apt-get install -y \
    ocrmypdf

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /home

# Copy only dependency files first (leverage cache when code changes but deps don't)
COPY pyproject.toml uv.lock ./

# Install dependencies (this layer gets cached unless deps change)
RUN uv sync --frozen --no-cache

# Copy tessdata files (separate from main code for better caching)
COPY tessdata/ tessdata/
RUN mv tessdata/ind.traineddata /usr/share/tesseract-ocr/5/tessdata/

# Copy the rest of the application code (this changes most frequently)
COPY . .

# Set the env path.
ENV PYTHONPATH="/home"

# Expose Port
EXPOSE 8000

CMD ["/home/.venv/bin/fastapi", "run", "file_extractor/app/extractor.py", "--port", "8000", "--host", "0.0.0.0"]