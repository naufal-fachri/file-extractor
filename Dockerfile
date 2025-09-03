FROM python:3.11-slim

# Install ocrmypdf and its dependencies.
RUN apt-get update && apt-get install -y \
    ocrmypdf

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /home

# Install the application dependencies.
COPY . .
RUN uv sync --frozen --no-cache

# Copy ind ind.tessdata to ./tessdata/
RUN mv tessdata/* /usr/share/tesseract-ocr/5/tessdata/

# Set the env path.
ENV PYTHONPATH="/home"

# Expose Port
EXPOSE 8000

CMD ["/home/.venv/bin/fastapi", "run", "file_extractor/app/extractor.py", "--port", "8000", "--host", "0.0.0.0"]