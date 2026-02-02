# GDrive to Vertex AI RAG Pipeline

A document Q&A pipeline that processes PDFs from Google Drive using Google Cloud's Document AI, creates vector embeddings, and provides an AI-powered question-answering interface using Vertex AI and the Agent Development Kit (ADK).

## Architecture

```
Google Drive PDF → Document AI (OCR) → Chunking → Embeddings → Vector Search → ADK Agent → Answers
```

### Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `phase1_ingestion.py` | Downloads PDF from Drive, uploads to GCS, runs Document AI OCR |
| 2 | `phase2_chunking.py` | Parses Document AI output into semantic chunks with metadata |
| 3 | `phase3_indexing.py` | Generates embeddings and creates Vertex AI Vector Search index |
| 4a | `phase4_deploy_index.py` | Deploys the index to an endpoint for querying |
| 4b | `phase4_adk.py` | Interactive Q&A interface using Gemini + ADK |

## Prerequisites

- Python 3.11+
- Google Cloud Project with billing enabled
- APIs enabled (see `REQUIRED_APIS_AND_PERMISSIONS.txt`):
  - Document AI API
  - Vertex AI API
  - Cloud Storage API
  - Google Drive API

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/utkarshdeo-collab/GDrive_Vertex_AI_Pipeline.git
cd GDrive_Vertex_AI_Pipeline

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Configure `config.py`

Update these values in `config.py`:

```python
PROJECT_ID = "your-project-id"
DOCAI_PROCESSOR_ID = "your-document-ai-processor-id"
DRIVE_FILE_ID = "your-google-drive-file-id"
```

## Running the Pipeline

Execute each phase in order:

```bash
# Phase 1: Ingest PDF from Google Drive
python phase1_ingestion.py

# Phase 2: Chunk the document
python phase2_chunking.py

# Phase 3: Create embeddings and Vector Search index
python phase3_indexing.py
# Note: Update VECTOR_INDEX_ID in config.py with the output

# Phase 4a: Deploy the index
python phase4_deploy_index.py
# Note: Update DEPLOYED_INDEX_ID in config.py if needed

# Phase 4b: Run the Q&A interface
python phase4_adk.py
```

## Usage

After running `phase4_adk.py`, enter your questions about the document:

```
Enter your question: What were the key findings in the report?
```

The system will:
1. Search for relevant passages using vector similarity
2. Pass context to Gemini model
3. Return an answer based on the document content

## Configuration Reference

| Variable | Description |
|----------|-------------|
| `PROJECT_ID` | Google Cloud project ID |
| `LOCATION` | Vertex AI region (default: us-central1) |
| `GCS_BUCKET_NAME` | Cloud Storage bucket for staging |
| `DOCAI_PROCESSOR_ID` | Document AI processor ID |
| `DRIVE_FILE_ID` | Google Drive file ID of the PDF |
| `VECTOR_INDEX_ID` | Vector Search index ID (from Phase 3) |
| `DEPLOYED_INDEX_ID` | Deployed index ID (from Phase 4a) |
| `GEMINI_MODEL` | Gemini model for Q&A (default: gemini-2.0-flash-001) |

## Utility Scripts

- `validate_apis.py` - Check if required APIs are enabled
- `check_gemini_models.py` - List available Gemini models in your project

## Tech Stack

- **Document Processing**: Google Document AI
- **Embeddings**: Vertex AI Text Embedding (text-embedding-004)
- **Vector Search**: Vertex AI Matching Engine
- **LLM**: Gemini 2.0 Flash
- **Agent Framework**: Google Agent Development Kit (ADK)
- **Storage**: Google Cloud Storage

## License

MIT
