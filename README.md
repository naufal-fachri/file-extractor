# 📂 File Extractor

File Extractor is a modular Python project for extracting and processing content from different file formats (e.g., **PDFs, Word documents**) with support for OCR using **Tesseract**.

---

## 🚀 Features

- 📑 **PDF extraction** using `pdf_extractor.py`
- 📝 **Word document extraction** using `word_extractor.py`
- 🔍 OCR support with **Tesseract** (via `tessdata`)
- 🧩 Modular architecture (`app` for core logic, `tools` for format-specific extractors)
- 🐳 Ready-to-use **Dockerfile** for containerized execution

---

## 📂 Project Structure

file_extractor/
│
├── app/ # Core application logic using FastAPI
│ ├── init.py
│ ├── extractor.py
│
├── tools/ # File-type specific extractors
│ ├── init.py
│ ├── pdf_extractor.py
│ ├── word_extractor.py
│
├── tessdata/ # OCR data files for Tesseract
│
├── .dockerignore # Files to ignore in Docker builds
├── .gitignore # Git ignored files
├── .python-version # Python version manager file
├── Dockerfile # Docker build instructions
├── pyproject.toml # Project dependencies & build config
├── README.md # Project documentation
├── uv.lock # Lock file for reproducible installs

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/file-extractor.git
cd file-extractor
```

### 2. Install dependencies (using uv)
```bash
uv sync --frozen
```

## 🐳 Run with Docker

Build the Docker image:
```bash
docker build -t file-extractor:{your version tag} .
```

Run the container:
```bash
docker run --rm -p {your port selection}:8000 file-extractor
```