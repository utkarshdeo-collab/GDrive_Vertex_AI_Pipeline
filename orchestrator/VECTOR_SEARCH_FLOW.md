# Vector Search Flow Documentation

## Overview

This document describes the vector-based flow for querying Salesforce and Domo data through Vertex AI Vector Search index.

## Architecture Flow

```
User
  │
  ▼
Orchestrator
  │
  │ get_salesforce_account(A123)
  ▼
Salesforce Agent
  │ build text: "Account A123"
  │ embed → vector
  │ query endpoint(filter=SF)
  ▼
Vector Endpoint (SF+Domo index)
  │
  ▼
Salesforce Agent → returns SF data
  │
  ▼
Orchestrator extracts POD45
  │
  │ get_domo_pod(POD45)
  ▼
Domo Agent
  │ build text: "POD45"
  │ embed → vector
  │ query endpoint(filter=Domo)
  ▼
Vector Endpoint (same index)
  │
  ▼
Domo Agent → returns Domo data
  │
  ▼
Orchestrator joins + computes
  │
  ▼
Nexus Snapshot
```

## Components

### 1. Vector Endpoint Module (`vector_endpoint.py`)

Shared module for querying the Vertex AI Vector Search index with filters.

**Key Functions:**
- `query_vector_index(query_text, filter_type, top_k)`: Query the index with optional filter
  - `filter_type`: "SF" for Salesforce, "Domo" for Domo, None for no filter
  - Returns vector search results with IDs and distances

### 2. Salesforce Agent (`salesforce_agent.py`)

**Flow:**
1. Receives account name (e.g., "A123")
2. Builds query text: "Account A123"
3. Calls `query_vector_index("Account A123", filter_type="SF")`
4. Extracts Salesforce data from results
5. Returns data including `pod_id` for Domo lookup

**Key Function:**
- `get_salesforce_account_data(account_name)`: Returns SF data + pod_id

### 3. Domo Agent (`domo_agent.py`)

**Flow:**
1. Receives pod_id (e.g., POD45)
2. Builds query text: "POD45"
3. Calls `query_vector_index("POD45", filter_type="Domo")`
4. Extracts Domo pod metrics from results
5. Returns pod data (MEAU, health_score, etc.)

**Key Function:**
- `get_pod_data_by_id(pod_id)`: Returns Domo pod metrics

### 4. Orchestrator (`run_orchestrator.py`)

Coordinates the flow:
1. Calls `get_salesforce_account_data()` → gets SF data + pod_id
2. Calls `get_domo_pod_data_by_id(pod_id)` → gets Domo metrics
3. Joins and computes → creates Nexus Snapshot

**Key Function:**
- `get_nexus_account_snapshot_orchestrated(account_name)`: Returns formatted Nexus Snapshot

## Setup Instructions

### Step 1: Index Salesforce and Domo Data

Run the indexing script to populate the vector index:

```bash
python orchestrator/index_sf_domo_data.py
```

This script:
- Fetches Salesforce data from `nexus_data.test_dataset2`
- Fetches Domo data from `domo_test_dataset.test_pod`
- Generates embeddings using `text-embedding-004`
- Uploads to GCS
- Creates/updates Vector Search index with namespace filters

### Step 2: Deploy Index to Endpoint

Deploy the index to a Vector Search endpoint:

```bash
python Drive_Ingestion_Phases/phase4_deploy_index.py
```

Or manually in Console:
1. Go to Vertex AI → Vector Search → Index Endpoints
2. Select/create an endpoint
3. Deploy the index created in Step 1

### Step 3: Update Configuration

Update `config.py`:
- `VECTOR_INDEX_ID`: The index ID from Step 1
- `INDEX_ENDPOINT_DISPLAY_NAME`: Endpoint display name
- `DEPLOYED_INDEX_ID`: Deployed index ID on the endpoint

### Step 4: Run Orchestrator

```bash
python orchestrator/run_orchestrator.py
```

## Vector Index Structure

The index stores data with namespace filters:

- **Salesforce entries**: `source=SF` or `source=Salesforce`
- **Domo entries**: `source=Domo`

Each entry contains:
- `id`: Unique identifier (e.g., `sf_0_001We00000UrUEpIAN` or `domo_0_pod45`)
- `embedding`: 768-dimensional vector from text-embedding-004
- `restricts`: Namespace filter for source type
- Metadata: Full data record (stored in chunks.jsonl)

## Query Flow Details

### Salesforce Query

1. **Input**: Account name "A123"
2. **Query Text**: "Account A123"
3. **Embedding**: Generated via `text-embedding-004`
4. **Filter**: `Namespace(name="source", allow_tokens=["SF", "Salesforce"])`
5. **Result**: Top-k similar Salesforce account records

### Domo Query

1. **Input**: pod_id = 45
2. **Query Text**: "POD45"
3. **Embedding**: Generated via `text-embedding-004`
4. **Filter**: `Namespace(name="source", allow_tokens=["Domo"])`
5. **Result**: Top-k similar Domo pod records

## Fallback Behavior

If vector search fails or returns no results, both agents fall back to BigQuery queries. This ensures reliability during the transition period.

## Benefits

1. **Unified Index**: Single vector index for both SF and Domo data
2. **Semantic Search**: Find accounts/pods by natural language queries
3. **Filtered Queries**: Namespace filters ensure correct data source
4. **Scalable**: Vector search handles large datasets efficiently
5. **Flexible**: Easy to add more data sources with new filters

## Future Enhancements

- Store full data in vector index metadata (eliminate BigQuery fallback)
- Add more data sources (e.g., Support tickets, Jira)
- Implement hybrid search (vector + keyword)
- Add query caching for performance
