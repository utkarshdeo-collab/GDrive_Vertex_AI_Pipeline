# Orchestrator — Master Agent + PDF & Salesforce Sub-Agents

Single Gemini model; master agent routes to PDF agent or Salesforce agent (or both) and synthesizes answers.

## Run

From project root:

```bash
python -m orchestrator.run_orchestrator
```

Or:

```bash
cd orchestrator
python run_orchestrator.py
```

## Prerequisites

- Phase 4 (Vector Search index) deployed so PDF search works.
- BigQuery dataset `nexus_data` and `Salesforce/nexus_schema.json` present.
- `config.py`: PROJECT_ID, LOCATION, GEMINI_MODEL set.

## Flow

1. User asks a question.
2. Master agent uses the routing data dictionary (nexus_schema summary) to decide: PDF or Salesforce.
3. Master delegates to exactly one sub-agent (pdf_agent or salesforce_agent).
4. Sub-agent runs (search_document or execute_sql).
5. Master returns the answer.

**Combined questions:** If the user asks about both document and Salesforce in one question, the master replies: "Please ask about the document or about Salesforce data separately."
