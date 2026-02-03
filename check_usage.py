"""
Google Cloud Resource Usage Checker for Document Pipeline.
Shows free tier usage, billable resources, and estimated costs.
"""

import config
from google.cloud import storage
from google.cloud import aiplatform
from datetime import datetime, timedelta

# ============== PRICING INFO (as of 2025) ==============
# These are approximate - check cloud.google.com/pricing for current rates

PRICING = {
    "storage_per_gb_month": 0.020,  # Standard storage $/GB/month
    "document_ai_per_page": 0.001,  # Document OCR $/page (after free tier)
    "embedding_per_1k_chars": 0.000025,  # text-embedding-004 $/1K characters
    "vector_search_per_node_hour": 0.10,  # Matching Engine $/node/hour (approximate)
    "gemini_flash_per_1m_input": 0.075,  # Gemini 2.0 Flash input $/1M tokens
    "gemini_flash_per_1m_output": 0.30,  # Gemini 2.0 Flash output $/1M tokens
}

FREE_TIER = {
    "document_ai_pages": 1000,  # 1000 pages/month free
    "storage_gb": 5,  # 5GB free (first 90 days)
    "gemini_flash_requests": 15,  # 15 RPM free tier
}


def format_bytes(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def check_storage_usage():
    """Check Cloud Storage usage."""
    print("\n" + "=" * 60)
    print("  CLOUD STORAGE USAGE")
    print("=" * 60)
    
    try:
        client = storage.Client(project=config.PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET_NAME)
        
        if not bucket.exists():
            print(f"  Bucket '{config.GCS_BUCKET_NAME}' not found")
            return 0
        
        total_size = 0
        blob_counts = {}
        
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            total_size += blob.size
            prefix = blob.name.split('/')[0] if '/' in blob.name else 'root'
            blob_counts[prefix] = blob_counts.get(prefix, 0) + 1
        
        total_gb = total_size / (1024 ** 3)
        monthly_cost = max(0, total_gb - FREE_TIER["storage_gb"]) * PRICING["storage_per_gb_month"]
        
        print(f"\n  Bucket: {config.GCS_BUCKET_NAME}")
        print(f"  Total Size: {format_bytes(total_size)} ({total_gb:.4f} GB)")
        print(f"  Total Objects: {len(blobs)}")
        print(f"\n  Breakdown by prefix:")
        for prefix, count in sorted(blob_counts.items()):
            print(f"    - {prefix}/: {count} files")
        
        print(f"\n  Free Tier: {FREE_TIER['storage_gb']} GB/month (first 90 days)")
        print(f"  Billable: {max(0, total_gb - FREE_TIER['storage_gb']):.4f} GB")
        print(f"  Estimated Monthly Cost: ${monthly_cost:.4f}")
        
        return total_size
        
    except Exception as e:
        print(f"  Error checking storage: {e}")
        return 0


def check_vector_search_resources():
    """Check Vertex AI Vector Search resources."""
    print("\n" + "=" * 60)
    print("  VERTEX AI VECTOR SEARCH RESOURCES")
    print("=" * 60)
    
    try:
        aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
        
        # Check indexes
        indexes = aiplatform.MatchingEngineIndex.list()
        print(f"\n  Indexes: {len(indexes)}")
        for idx in indexes:
            print(f"    - {idx.display_name}")
            print(f"      Resource: {idx.resource_name}")
            print(f"      Created: {idx.create_time}")
        
        # Check endpoints
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
        print(f"\n  Index Endpoints: {len(endpoints)}")
        
        deployed_count = 0
        for ep in endpoints:
            print(f"    - {ep.display_name}")
            print(f"      Resource: {ep.resource_name}")
            deployed = ep.deployed_indexes or []
            deployed_count += len(deployed)
            for d in deployed:
                print(f"      Deployed: {d.id}")
        
        # Estimate costs (Vector Search charges per node-hour when deployed)
        if deployed_count > 0:
            hours_per_month = 24 * 30
            # Minimum 2 nodes for high availability
            estimated_monthly = 2 * hours_per_month * PRICING["vector_search_per_node_hour"]
            print(f"\n  ⚠️  DEPLOYED INDEXES ARE BILLABLE!")
            print(f"  Deployed Index Count: {deployed_count}")
            print(f"  Estimated Monthly Cost: ${estimated_monthly:.2f} (2 nodes minimum)")
            print(f"  💡 Tip: Undeploy indexes when not in use to save costs")
        else:
            print(f"\n  ✅ No deployed indexes (no ongoing charges)")
        
        return deployed_count
        
    except Exception as e:
        print(f"  Error checking Vector Search: {e}")
        return 0


def check_document_ai_usage():
    """Check Document AI processor info."""
    print("\n" + "=" * 60)
    print("  DOCUMENT AI")
    print("=" * 60)
    
    try:
        from google.cloud import documentai_v1 as documentai
        
        client = documentai.DocumentProcessorServiceClient()
        parent = f"projects/{config.PROJECT_ID}/locations/{config.DOCAI_LOCATION}"
        
        processors = list(client.list_processors(parent=parent))
        print(f"\n  Processors: {len(processors)}")
        for proc in processors:
            print(f"    - {proc.display_name}")
            print(f"      Type: {proc.type_}")
            print(f"      State: {proc.state.name}")
        
        print(f"\n  Free Tier: {FREE_TIER['document_ai_pages']} pages/month")
        print(f"  After Free Tier: ${PRICING['document_ai_per_page']}/page")
        print(f"\n  Note: Exact page count requires Billing API access")
        
    except Exception as e:
        print(f"  Error checking Document AI: {e}")


def check_embedding_usage():
    """Show embedding model info."""
    print("\n" + "=" * 60)
    print("  TEXT EMBEDDING API")
    print("=" * 60)
    
    print(f"\n  Model: {config.EMBEDDING_MODEL}")
    print(f"  Pricing: ${PRICING['embedding_per_1k_chars']} per 1K characters")
    print(f"\n  Your Pipeline Usage (estimated):")
    
    # Estimate based on chunks
    try:
        client = storage.Client(project=config.PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET_NAME)
        chunks_blob = bucket.blob(f"{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl")
        
        if chunks_blob.exists():
            data = chunks_blob.download_as_text()
            lines = data.strip().split('\n')
            total_chars = len(data)
            
            # Phase 3 embedding cost
            embedding_cost = (total_chars / 1000) * PRICING['embedding_per_1k_chars']
            
            print(f"    - Chunks indexed: {len(lines)}")
            print(f"    - Total characters: {total_chars:,}")
            print(f"    - Phase 3 embedding cost: ${embedding_cost:.4f}")
            
            # Phase 4 query cost estimate
            avg_queries_per_session = 10
            avg_query_length = 200  # characters
            query_cost_per_session = (avg_queries_per_session * avg_query_length / 1000) * PRICING['embedding_per_1k_chars']
            print(f"    - Per Q&A session (~10 queries): ${query_cost_per_session:.6f}")
        else:
            print(f"    - No chunks found (Phase 2 not run yet)")
            
    except Exception as e:
        print(f"  Error estimating embedding usage: {e}")


def check_gemini_usage():
    """Show Gemini model info."""
    print("\n" + "=" * 60)
    print("  GEMINI API")
    print("=" * 60)
    
    print(f"\n  Model: {config.GEMINI_MODEL}")
    print(f"  Input Pricing: ${PRICING['gemini_flash_per_1m_input']} per 1M tokens")
    print(f"  Output Pricing: ${PRICING['gemini_flash_per_1m_output']} per 1M tokens")
    print(f"\n  Estimated per Q&A query:")
    
    # Estimate based on typical usage
    avg_context_tokens = 3000  # ~12K chars of context
    avg_output_tokens = 500    # typical response
    
    input_cost = (avg_context_tokens / 1_000_000) * PRICING['gemini_flash_per_1m_input']
    output_cost = (avg_output_tokens / 1_000_000) * PRICING['gemini_flash_per_1m_output']
    
    print(f"    - Context: ~{avg_context_tokens} tokens → ${input_cost:.6f}")
    print(f"    - Response: ~{avg_output_tokens} tokens → ${output_cost:.6f}")
    print(f"    - Total per query: ${input_cost + output_cost:.6f}")
    print(f"    - 100 queries: ${(input_cost + output_cost) * 100:.4f}")


def print_summary():
    """Print cost summary."""
    print("\n" + "=" * 60)
    print("  💰 COST SUMMARY")
    print("=" * 60)
    
    print("""
  ONE-TIME COSTS (already incurred):
  ┌─────────────────────────────────────────────────────┐
  │ Document AI OCR        ~$0.00 - $0.05 (free tier)   │
  │ Embedding Generation   ~$0.01 - $0.05              │
  │ Index Creation         ~$0.00 (no charge)          │
  └─────────────────────────────────────────────────────┘
  
  ONGOING COSTS (while resources exist):
  ┌─────────────────────────────────────────────────────┐
  │ Cloud Storage          ~$0.01/month (minimal)       │
  │ ⚠️  DEPLOYED INDEX      ~$144/month (if deployed!)  │
  │ Per Q&A Query          ~$0.0001 - $0.001           │
  └─────────────────────────────────────────────────────┘
  
  ⚠️  IMPORTANT: Deployed Vector Search indexes charge
     ~$0.10/node/hour. With minimum 2 nodes = $144/month!
     
  💡 To stop charges: Undeploy the index when not testing.
""")


def main():
    print("\n" + "=" * 60)
    print("  GOOGLE CLOUD RESOURCE USAGE CHECKER")
    print(f"  Project: {config.PROJECT_ID}")
    print(f"  Location: {config.LOCATION}")
    print(f"  Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    check_storage_usage()
    check_vector_search_resources()
    check_document_ai_usage()
    check_embedding_usage()
    check_gemini_usage()
    print_summary()


if __name__ == "__main__":
    main()
