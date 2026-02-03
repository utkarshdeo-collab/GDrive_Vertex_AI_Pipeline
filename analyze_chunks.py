"""
Analyze chunks before embedding to ensure quality.
Checks for potential issues that could cause poor RAG answers.
"""
import json
import sys
import re
from collections import Counter

import config
from google.cloud import storage


def load_chunks_from_gcs():
    """Load chunks from GCS."""
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
    blob = bucket.blob(f"{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl")
    
    if not blob.exists():
        print("ERROR: chunks.jsonl not found in GCS. Run Phase 2 first.")
        sys.exit(1)
    
    data = blob.download_as_text()
    chunks = []
    for line in data.strip().split("\n"):
        if line:
            chunks.append(json.loads(line))
    return chunks


def analyze_chunks(chunks):
    """Analyze chunk quality and potential issues."""
    print("\n" + "=" * 60)
    print("  CHUNK ANALYSIS REPORT")
    print("=" * 60)
    
    # Basic stats
    total = len(chunks)
    table_chunks = [c for c in chunks if c["metadata"].get("is_table")]
    text_chunks = [c for c in chunks if not c["metadata"].get("is_table")]
    
    print(f"\n[1] BASIC STATISTICS")
    print(f"    Total chunks: {total}")
    print(f"    Table chunks: {len(table_chunks)}")
    print(f"    Text chunks: {len(text_chunks)}")
    
    # Character length analysis
    lengths = [len(c["text"]) for c in chunks]
    print(f"\n[2] CHUNK SIZE DISTRIBUTION")
    print(f"    Min length: {min(lengths)} chars")
    print(f"    Max length: {max(lengths)} chars")
    print(f"    Avg length: {sum(lengths) // len(lengths)} chars")
    
    # Size buckets
    tiny = sum(1 for l in lengths if l < 100)
    small = sum(1 for l in lengths if 100 <= l < 300)
    medium = sum(1 for l in lengths if 300 <= l < 1000)
    large = sum(1 for l in lengths if 1000 <= l < 3000)
    xlarge = sum(1 for l in lengths if l >= 3000)
    
    print(f"    < 100 chars (tiny):     {tiny} chunks")
    print(f"    100-300 chars (small):  {small} chunks")
    print(f"    300-1000 chars (medium): {medium} chunks")
    print(f"    1000-3000 chars (large): {large} chunks")
    print(f"    > 3000 chars (xlarge):  {xlarge} chunks")
    
    # Page distribution
    pages = [c["metadata"].get("page", 0) for c in chunks]
    page_counts = Counter(pages)
    print(f"\n[3] PAGE DISTRIBUTION")
    print(f"    Pages covered: {min(pages)} to {max(pages)}")
    print(f"    Chunks per page (avg): {total / len(page_counts):.1f}")
    
    # Check for potential issues
    print(f"\n[4] POTENTIAL ISSUES")
    issues = []
    
    # Issue: Very short chunks (might be fragmented)
    very_short = [c for c in chunks if len(c["text"]) < 50]
    if very_short:
        issues.append(f"⚠️  {len(very_short)} chunks are very short (<50 chars)")
        for c in very_short[:3]:
            preview = c["text"][:80].replace("\n", " ")
            print(f"       Example: '{preview}...'")
    
    # Issue: Tables without proper structure
    bad_tables = []
    for c in table_chunks:
        text = c["text"]
        # A good table should have multiple | characters
        pipe_count = text.count("|")
        if pipe_count < 4:
            bad_tables.append(c)
    if bad_tables:
        issues.append(f"⚠️  {len(bad_tables)} table chunks may be malformed")
    
    # Issue: Missing expected content (spot check)
    all_text = " ".join(c["text"] for c in chunks).lower()
    
    expected_terms = [
        ("numbers/currency", r"\$[\d,]+|\d+%|\d+\.\d+"),
        ("table markers", r"\|.*\|"),
    ]
    
    for name, pattern in expected_terms:
        matches = len(re.findall(pattern, all_text))
        print(f"    {name}: {matches} occurrences")
    
    if not issues:
        print("    ✅ No major issues detected")
    else:
        for issue in issues:
            print(f"    {issue}")
    
    # Sample table chunks
    print(f"\n[5] SAMPLE TABLE CHUNKS")
    for i, chunk in enumerate(table_chunks[:3]):
        print(f"\n    --- Table {i+1} (Page {chunk['metadata'].get('page')}) ---")
        lines = chunk["text"].split("\n")[:8]  # First 8 lines
        for line in lines:
            print(f"    {line[:80]}")
        if len(chunk["text"].split("\n")) > 8:
            print(f"    ... ({len(chunk['text'])} chars total)")
    
    # Sample text chunks
    print(f"\n[6] SAMPLE TEXT CHUNKS")
    for i, chunk in enumerate(text_chunks[:3]):
        print(f"\n    --- Text {i+1} (Page {chunk['metadata'].get('page')}) ---")
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"    {preview}...")
    
    # Search for specific content to verify completeness
    print(f"\n[7] CONTENT VERIFICATION")
    
    # Look for key terms that should be in a complete document
    search_terms = [
        "executive summary",
        "implementation",
        "cost",
        "results",
        "conclusion",
        "recommendation",
    ]
    
    for term in search_terms:
        found = sum(1 for c in chunks if term in c["text"].lower())
        status = "✅" if found > 0 else "❌"
        print(f"    {status} '{term}': found in {found} chunks")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    quality_score = 100
    if tiny > 5:
        quality_score -= 10
        print("  ⚠️  Many tiny chunks - may cause fragmented answers")
    if len(table_chunks) == 0:
        quality_score -= 20
        print("  ⚠️  No table chunks - tables may not be extracted")
    if bad_tables:
        quality_score -= 10
        print("  ⚠️  Some tables may be malformed")
    
    if quality_score >= 80:
        print(f"\n  ✅ QUALITY SCORE: {quality_score}/100 - Good to proceed!")
    elif quality_score >= 60:
        print(f"\n  ⚠️  QUALITY SCORE: {quality_score}/100 - Acceptable, may have some issues")
    else:
        print(f"\n  ❌ QUALITY SCORE: {quality_score}/100 - Consider re-chunking")
    
    print("=" * 60 + "\n")
    
    return quality_score >= 60


def main():
    print("\n  Loading chunks from GCS...")
    chunks = load_chunks_from_gcs()
    print(f"  Loaded {len(chunks)} chunks")
    
    ok = analyze_chunks(chunks)
    
    if ok:
        print("  Ready for Phase 3: python phase3_indexing.py\n")
    else:
        print("  Consider reviewing chunking before proceeding.\n")


if __name__ == "__main__":
    main()
