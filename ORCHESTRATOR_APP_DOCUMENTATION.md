# Orchestrator App: Document & Salesforce Q&A — Technical Overview

This document describes the **Orchestrator App** (orchestrator_app): a single entry point that answers questions about **uploaded documents (PDF)** and **Salesforce data in BigQuery**. It explains the steps involved in embeddings, Vector Search, and the flow from Salesforce API data into BigQuery, and how the orchestrator routes and answers user questions.

---

## 1. What Is the Orchestrator App?

The Orchestrator App is a **master agent** that:

- Accepts a user question in natural language (e.g. from a terminal or future UI).
- **Routes** the question to exactly one specialist:
  - **PDF agent** — for questions about uploaded documents (reports, case studies, implementation details, etc.).
  - **Salesforce agent** — for questions about Salesforce data (ARR, opportunities, customers, accounts, etc.) that lives in **BigQuery**.
- Returns the specialist’s answer. If the user asks about both document and Salesforce in one question, the master asks them to ask separately.

All agents use a single Gemini model (e.g. Gemini 2.0 Flash) and the same Google Cloud project and region. The app also shows a **per-question cost breakdown** in the terminal (Gemini tokens, embedding usage, BigQuery bytes processed).

---

## 2. Document / PDF Flow: From PDF to Answer (Embeddings & Vector Search)

This section describes how document content is prepared and how the PDF agent uses **embeddings** and **Vector Search** to answer questions.

### 2.1 Ingestion (Phase 1)

- **Input:** A PDF file (e.g. hospital implementation report, case study). The PDF can be provided as a **local file** or staged in **Google Cloud Storage (GCS)** in a project bucket.
- **Optional:** Document AI can be used for OCR or extra processing; it can be skipped and the pipeline can work from a locally parsed PDF.
- **Output:** The PDF is available in GCS (or locally) for the next phase.

### 2.2 Chunking (Phase 2)

- **Purpose:** Split the document into **chunks** of text that are small enough to embed and retrieve, but large enough to keep context (e.g. tables, sections).
- **How:** A chunking step (e.g. using **pdfplumber**) parses the PDF and produces text chunks, often with section headers so that “Financial Benefits” or “Lessons Learned” stay together. Tables are kept as coherent blocks (e.g. Markdown) instead of being split across chunks.
- **Output:** A **chunks file** (e.g. `chunks.jsonl`) stored in GCS. Each line is a chunk with an **ID** and **text**. This file is the source of truth for “what the document says” at query time.

### 2.3 Embeddings & Vector Index (Phase 3)

- **Embeddings:** Each chunk’s text is sent to an **embedding model** (e.g. Vertex AI **text-embedding-004**). The model returns a **vector** (a list of numbers) that represents the meaning of that chunk. Similar content (e.g. “billing accuracy,” “staff resistance”) gets similar vectors.
- **Vector Search index:** These vectors are written into a **Vertex AI Vector Search** index. The index stores:
  - Chunk ID (linking back to the chunks file in GCS).
  - The embedding vector.
  - Optional metadata (e.g. source = “doc-pipeline”) for filtering.
- **Output:** A **Vector Search index** in the project (e.g. in region us-east4). This index is later **deployed** to an **index endpoint** so that queries can run against it.

### 2.4 Deploying the Index (Phase 4 – Deploy)

- The Vector Search index is **deployed** to an **index endpoint** (e.g. “doc-pipeline-endpoint”) with a **deployed index ID** (e.g. “doc_pipeline_deployed”).
- Until this step is done, the PDF agent cannot run search; the orchestrator app assumes the endpoint and deployed index already exist.

### 2.5 Query Time: How the PDF Agent Answers

When the user asks a **document question** and the master routes to the **PDF agent**:

1. **Routing:** The master agent (orchestrator) calls Gemini once to decide: “This question is about the document → delegate to pdf_agent.”
2. **PDF agent decides to search:** The PDF agent (Gemini) sees the question and decides to use the **search_document** tool.
3. **search_document tool:**
   - Loads the **chunks file** from GCS (chunk ID → text).
   - Takes the user’s question (or a refined query) and sends it to the **same embedding model** (text-embedding-004) to get a **query vector**.
   - Calls **Vector Search** on the deployed index: “find the nearest neighbor vectors to this query vector” (e.g. top 50), with optional filter (e.g. source = doc-pipeline).
   - For each neighbor, gets the chunk ID and then the **text** from the chunks file.
   - Concatenates these texts into a **context** (up to a character limit) and returns it to the PDF agent.
4. **PDF agent answers:** Gemini receives the context (retrieved passages) and generates an answer **only from that context**, quoting numbers and facts from the document.

So the **steps involved in embeddings** for the document flow are:

- **Offline:** Embed every chunk → store vectors in Vector Search index → deploy index.
- **Online (per question):** Embed the user query → Vector Search finds similar chunks → retrieve chunk text from GCS → pass to Gemini as context for the answer.

---

## 3. Salesforce Data & BigQuery Flow

This section describes how **Salesforce data** gets into **BigQuery** and how the **Salesforce agent** uses it.

### 3.1 Getting Salesforce Data Into BigQuery

- **Source:** Data is in **Salesforce** (e.g. Accounts, Opportunities, custom objects). This data is synced or exported into **Google BigQuery** so that it can be queried with SQL.
- **Mechanism:** Typically this is done using **Salesforce APIs** (e.g. Bulk API or REST API) or an ETL/integration tool (e.g. Dataform, Fivetran, or a custom job) that:
  - Reads from Salesforce (e.g. accounts, opportunities, products).
  - Transforms and loads the data into BigQuery tables in a dataset (e.g. **nexus_data**).
- **Result:** BigQuery holds tables such as `sf_account_info`, `sf_opportunity_detailed`, `sf_portfolio_summary`, etc., with columns like Customer_Name, Total_ARR, Opportunity_Name, CloseDate, Contracted_Licenses, Stage. The schema (tables and columns) is documented in a file such as **nexus_schema.json**.

So the flow is: **Salesforce (source) → Salesforce API / ETL → BigQuery dataset (e.g. nexus_data)**. The Orchestrator App does not perform the sync; it assumes the data is already in BigQuery.

### 3.2 Schema: nexus_schema.json

- **nexus_schema.json** describes the BigQuery dataset and tables (table names, column names, data types). A copy lives in the **orchestrator** folder so that:
  - The **master agent** can use a short “routing” summary (table/column names) to decide: “This question is about Salesforce/BigQuery → delegate to salesforce_agent.”
  - The **Salesforce agent** gets the full schema in its instructions so it can generate the right SQL (e.g. `project_id.nexus_data.sf_account_info`).

### 3.3 Query Time: How the Salesforce Agent Answers

When the user asks a **Salesforce question** (e.g. “What’s the total ARR for Antino Bank?”) and the master routes to the **Salesforce agent**:

1. **Routing:** The master agent calls Gemini once to decide: “This question is about Salesforce/BigQuery data → delegate to salesforce_agent.”
2. **Salesforce agent:** Gemini (salesforce_agent) has the schema in its instructions and has access to an **execute_sql** tool that runs **read-only** SQL (SELECT only) in BigQuery.
3. **execute_sql:** The agent formulates a SQL query (e.g. `SELECT Total_ARR, Customer_Name FROM project_id.nexus_data.sf_account_info WHERE Customer_Name = 'Antino Bank'`). The tool runs the query in BigQuery, returns rows (and optionally bytes processed for cost), and the agent summarizes the answer for the user.

So the flow is: **User question → Master routes to salesforce_agent → salesforce_agent (Gemini) + execute_sql (BigQuery) → answer.**

---

## 4. End-to-End Orchestrator Flow

1. **User** types a question (e.g. in the terminal running `python run_orchestrator.py`).
2. **Optional routing hint:** If the question clearly mentions document topics (e.g. “billing accuracy,” “lessons learned”) or Salesforce topics (e.g. “ARR,” “opportunities”), a hint can be prepended so the master is more likely to choose the right agent.
3. **Master agent (orchestrator):** Receives the question (and hint if any). Calls Gemini once with a **routing data dictionary** (summary of BigQuery tables/columns from nexus_schema.json) and rules. It decides: **pdf_agent** or **salesforce_agent** (or “ask separately” if both are asked in one question).
4. **Delegation:** The master delegates to exactly one sub-agent. That sub-agent runs (possibly multiple tool calls: search_document one or more times, or execute_sql one or more times).
5. **Answer:** The sub-agent’s final answer is returned to the user.
6. **Cost breakdown:** After each question, the app prints a **cost for this question** (orchestrator Gemini, pdf_agent or salesforce_agent Gemini, search_document embedding chars, execute_sql BigQuery bytes, and total) using public list prices.

---

## 5. Summary Table

| Topic | What happens |
|-------|----------------|
| **Document content** | PDF → Chunking (e.g. pdfplumber) → Chunks in GCS → Embeddings (text-embedding-004) → Vector Search index → Deploy to endpoint. |
| **Document question** | User question → Master routes to pdf_agent → search_document (embed query, Vector Search, get chunk text from GCS) → Gemini answers from context. |
| **Salesforce data** | Salesforce (API) → ETL/sync → BigQuery dataset (e.g. nexus_data). Schema in nexus_schema.json. |
| **Salesforce question** | User question → Master routes to salesforce_agent → execute_sql (BigQuery SELECT) → Gemini summarizes rows. |
| **Orchestrator** | Single entry point; master routes to pdf_agent or salesforce_agent; per-question cost shown in terminal. |

---

*This document reflects the Orchestrator App as implemented with run_orchestrator.py, pdf_agent (document/Vector Search), and salesforce_agent (BigQuery execute_sql), using config such as PROJECT_ID, LOCATION, GEMINI_MODEL, and the nexus_schema in the orchestrator folder.*
