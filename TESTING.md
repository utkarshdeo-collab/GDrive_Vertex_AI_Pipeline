# How to Test the Orchestrator and Agents

## 1. Prerequisites

### 1.1 Environment and dependencies

From the pipeline root (`GDrive_Vertex_AI_Pipeline/`):

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# BigQuery is used by Salesforce and Domo agents (often pulled in via google-cloud-aiplatform)
pip install google-cloud-bigquery
```

### 1.2 Google Cloud authentication (ADC)

Use **Application Default Credentials** so the app can call Vertex AI and BigQuery:

```bash
# Log in (opens browser)
gcloud auth application-default login

# Set the project
gcloud config set project sym-dev-mr-agents-01
```

Optional: override project/region via environment variables:

```bash
set GCP_PROJECT_ID=sym-dev-mr-agents-01
set GCP_LOCATION=us-east4
```

### 1.3 Data and config

- **PDF agent**: Vector Search index must be deployed (Phase 4). If not, PDF questions will fail; you can still test Salesforce and Domo.
- **Salesforce agent**: BigQuery dataset `nexus_data` must exist; `orchestrator/nexus_schema.json` should match your tables.
- **Domo agent**: BigQuery dataset `domo_test_dataset` must exist; update `orchestrator/domo_schema.json` and `Domo/domo_schema.json` with your real table/column names.

---

## 2. Test the Domo agent alone (local)

Tests only the Domo sub-agent and BigQuery, without the orchestrator.

**From project root** (`GDrive_Vertex_AI_Pipeline/`):

```bash
python Domo/run_agent_local.py
```

- Uses ADC for auth.
- Reads schema from `Domo/domo_schema.json`.
- Queries `project_id.domo_test_dataset.*`.

**Example questions:**

- “How many rows are in the domo_test_dataset?”
- “Show me a sample of data from domo_test_dataset.”  
  (Adjust question to match your actual table names in the schema.)

Type `exit` or `quit` to stop.

---

## 3. Test the Salesforce agent alone (local)

Same idea as Domo, for the Salesforce sub-agent:

```bash
python Salesforce/run_agent_local.py
```

**Example questions:**

- “What is the total ARR for Antino Bank?”
- “List customers in nexus_data.”
- “How many opportunities are there?”

Type `exit` or `quit` to stop.

---

## 4. Test the full orchestrator (all three agents)

Runs the **master agent** that routes to PDF, Salesforce, or Domo.

**From project root** (`GDrive_Vertex_AI_Pipeline/`):

```bash
python -m orchestrator.run_orchestrator
```

Or:

```bash
cd orchestrator
python run_orchestrator.py
```

You should see something like:

```
  ORCHESTRATOR — PDF + Salesforce + Domo (Master + 3 Sub-Agents)
  Project: sym-dev-mr-agents-01
  Region:  us-east4
  Model:   gemini-2.0-flash-001
  Ready. Ask about documents (PDF), Salesforce/BigQuery data, or Domo/BigQuery data.
  Type 'exit' to quit.
```

### Example questions by agent

**Salesforce (nexus_data):**

- “What’s the total ARR for Antino Bank?”
- “List opportunities from Salesforce.”
- “How many customers do we have in nexus_data?”

**Domo (domo_test_dataset):**

- “Query the domo_test_dataset and show me a summary.”
- “What’s in the Domo dataset?”
- “How many rows are in domo_test_dataset?”

**PDF / document:**

- “What are the key lessons learned in the document?”
- “What’s the implementation cost in the report?”
- “Summarize the executive summary from the PDF.”

After each answer, the app prints a **cost breakdown** (orchestrator + sub-agent Gemini, embeddings if PDF, BigQuery bytes if Salesforce/Domo).

Type `exit` or `quit` to exit.

---

## 5. Quick checklist

| Step | Command / check |
|------|------------------|
| Auth | `gcloud auth application-default login` |
| Project | `gcloud config set project sym-dev-mr-agents-01` |
| Deps | `pip install -r requirements.txt google-cloud-bigquery` |
| Domo only | `python Domo/run_agent_local.py` |
| Salesforce only | `python Salesforce/run_agent_local.py` |
| Full app | `python -m orchestrator.run_orchestrator` |

---

## 6. Troubleshooting

- **“Authentication failed”**  
  Run `gcloud auth application-default login` and ensure the project is set.

- **“Permission denied” / 403**  
  Your ADC identity needs roles such as: Vertex AI User, BigQuery Data Viewer (or Job User) on the datasets, and (for PDF) access to the Vector Search index.

- **PDF agent fails or “index not found”**  
  Complete Phase 4 (build and deploy the Vector Search index). Until then, only test with Salesforce or Domo questions.

- **Domo agent: “Table not found” or wrong columns**  
  Update `orchestrator/domo_schema.json` and `Domo/domo_schema.json` with the real `domo_test_dataset` tables and columns.

- **Wrong agent answers (e.g. Domo question goes to Salesforce)**  
  Use explicit wording (“domo”, “domo_test_dataset”) or add more routing keywords in `run_orchestrator.py` (`_DOMO_KEYWORDS`, etc.).

- **Import errors**  
  Run from `GDrive_Vertex_AI_Pipeline/` so `config` and `orchestrator` resolve correctly, or ensure the project root is on `PYTHONPATH`.
