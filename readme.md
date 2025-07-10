# RAG System for E-commerce: Setup and Usage Guide

A Retrieval-Augmented Generation (RAG) system implementation for e-commerce applications. This system enhances language model responses with relevant information retrieved from your product database.

**Example Application**: [Cafés 1808](https://www.cafes1808.com/)


## Prerequisites

Before you begin, ensure you have:
- Python 3.12.3 or higher installed
- pip package manager

## Setup Instructions

### Virtual Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv env

# Activate environment (Linux/macOS)
source env/bin/activate

# Activate environment (Windows)
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Add Your E-commerce Data
To enable the RAG agent to provide accurate customer assistance, please add your product data following these guidelines:

#### Data Collection Requirements
Scrape or prepare all necessary product information including:

 - How to Add Your Data
 - Locate the template in:
    ```bash
        data/raw-data/template.md
    ```

 - Save files in:
     ```bash
        data/
    ```

### Create `.env` File
Create your **`.env` file** using `.env.example` as a template, and fill in your specific values.

## Execution
After adding your data.md file.

1. Run the processing pipeline:
   ```bash
   python model_pipeline.py
2. Start the query agent:
   ```bash
   python query_executor.py

# Start Application
   - From the root folder.
   ```bash
   PYTHONPATH=$(pwd) streamlit run src/ui/app.py
