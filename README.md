# Medical Bill Extraction Pipeline

A FastAPI-based solution for extracting line items from medical bills and invoices using AI-powered document processing.

## Features

- 📄 Multi-format document support (Images, PDFs)
- 🔍 AI-powered line item extraction
- 💰 Automatic total reconciliation
- 🚫 Duplicate detection and removal
- 🎯 Confidence-based fallback processing
- 🔧 Image preprocessing for better OCR

## Quick Start

### Prerequisites

- Python 3.8+
- VS Code (recommended)

### Installation

1. **Clone and setup project in VS Code:**
   ```bash
   mkdir bill-extraction-pipeline
   cd bill-extraction-pipeline
   code .

2.**Create virtual environment:**
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On Mac/Linux:
  source venv/bin/activate