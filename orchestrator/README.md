# Orchestrator — Master Agent + PDF, Salesforce & Domo Sub-Agents

Single Gemini model; master agent routes to **pdf_agent**, **salesforce_agent**, or **domo_agent** and returns that specialist’s answer.

## Run

From project root (`GDrive_Vertex_AI_Pipeline/`):

```bash
python -m orchestrator.run_orchestrator
```

Or:

```bash
cd orchestrator
python run_orchestrator.py
```

## Prerequisites

- **Auth:** `gcloud auth application-default login` and project set.
- **PDF agent:** Phase 4 (Vector Search index) deployed.
- **Salesforce agent:** BigQuery dataset `nexus_data` and `orchestrator/nexus_schema.json`.
- **Domo agent:** BigQuery dataset `domo_test_dataset` and `orchestrator/domo_schema.json`.
- `config.py`: PROJECT_ID, LOCATION, GEMINI_MODEL set (or use env: GCP_PROJECT_ID, GCP_LOCATION).

## Flow

1. User asks a question.
2. Optional routing hint is added from keywords (Salesforce / Domo / document).
3. Master agent uses the routing data dictionary (schema summaries) to decide: PDF, Salesforce, or Domo.
4. Master delegates to exactly one sub-agent (pdf_agent, salesforce_agent, or domo_agent).
5. Sub-agent runs (search_document or execute_sql).
6. Master returns the answer and a per-question cost breakdown.

**Combined questions:** If the user asks about more than one source (document + Salesforce + Domo) in one question, the master replies: "Please ask about the document, Salesforce data, or Domo data separately."

## Testing

See **[TESTING.md](../TESTING.md)** in the project root for step-by-step testing (Domo-only, Salesforce-only, and full orchestrator) and troubleshooting.
