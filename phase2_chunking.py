"""
Phase 2: Smart Semantic Chunking (Local PDF Parsing)

Uses pdfplumber to parse PDF locally (no Document AI needed).
- Extracts tables as complete Markdown blocks (preserves structure)
- Detects section headers for context
- Creates semantic chunks with metadata

Output: JSONL of chunks uploaded to gs://<bucket>/chunks/chunks.jsonl
"""
import json
import re
import sys
import os

import pdfplumber
import config
from google.cloud import storage


def extract_table_as_markdown(table):
    """Convert pdfplumber table to Markdown format."""
    if not table or not table.extract():
        return ""
    
    rows = table.extract()
    if not rows:
        return ""
    
    # Clean cells: replace None with empty string, strip whitespace
    cleaned_rows = []
    for row in rows:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        # Skip completely empty rows
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
    
    if not cleaned_rows:
        return ""
    
    # Build markdown table
    md_lines = []
    
    # First row as header
    header = cleaned_rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # Remaining rows as body
    for row in cleaned_rows[1:]:
        # Pad row if needed
        while len(row) < len(header):
            row.append("")
        md_lines.append("| " + " | ".join(row[:len(header)]) + " |")
    
    return "\n".join(md_lines)


def is_section_header(text):
    """Detect if text is a section header."""
    if not text:
        return False
    
    text = text.strip()
    
    # Empty or too long
    if not text or len(text) > 150:
        return False
    
    # Numbered sections: "1.2 Title", "3.1.1 Subtitle"
    if re.match(r"^\d+(\.\d+)*\.?\s+\w", text):
        return True
    
    # "Table N:" or "Figure N:"
    if re.match(r"^(Table|Figure|Chart|Appendix)\s+\d+", text, re.I):
        return True
    
    # All caps short text (likely a header)
    if text.isupper() and len(text) < 80:
        return True
    
    # Short text without period at end (likely a title)
    if len(text) < 80 and not text.endswith('.') and not text.endswith(':'):
        # Check if it has title-like capitalization
        words = text.split()
        if len(words) <= 10 and words[0][0].isupper():
            return True
    
    return False


def extract_text_blocks(page, page_num):
    """Extract text from a page, grouping into logical blocks."""
    text = page.extract_text() or ""
    if not text.strip():
        return []
    
    # Split by double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    blocks = []
    for para in paragraphs:
        para = para.strip()
        if para and len(para) > 10:  # Skip very short fragments
            blocks.append(para)
    
    return blocks


def chunk_pdf(pdf_path, doc_label="doc0"):
    """
    Parse PDF and create chunks with proper table handling and section context.
    
    Returns list of {"id", "text", "metadata"}
    """
    chunks = []
    current_section = "Document"
    
    print(f"   Opening PDF: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Total pages: {total_pages}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num % 10 == 0:
                print(f"   Processing page {page_num}/{total_pages}...")
            
            # Extract tables first (so we can exclude table regions from text)
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables] if tables else []
            
            # Extract and chunk tables
            for tbl_idx, table in enumerate(tables):
                table_md = extract_table_as_markdown(table)
                if table_md and len(table_md) > 20:
                    chunk_id = f"{doc_label}_p{page_num}_tbl{tbl_idx}"
                    
                    # Try to get table title from text just above table
                    table_context = current_section
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": f"[Page {page_num} | Section: {table_context}]\n\n{table_md}",
                        "metadata": {
                            "type": "table",
                            "page": page_num,
                            "section_header": table_context,
                            "is_table": True,
                        },
                    })
            
            # Extract text blocks (excluding table regions would require more complex logic)
            text_blocks = extract_text_blocks(page, page_num)
            
            for blk_idx, block in enumerate(text_blocks):
                # Update section header if this looks like one
                if is_section_header(block):
                    current_section = block[:100]  # Limit length
                
                chunk_id = f"{doc_label}_p{page_num}_blk{blk_idx}"
                
                # Create chunk with context
                chunk_text = f"[Page {page_num} | Section: {current_section}]\n\n{block}"
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "type": "text",
                        "page": page_num,
                        "section_header": current_section,
                        "is_table": False,
                    },
                })
    
    return chunks


def merge_small_chunks(chunks, min_chars=200):
    """Merge very small consecutive chunks from the same page."""
    if not chunks:
        return chunks
    
    merged = []
    buffer = None
    
    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue
        
        # If current buffer is too small and same page, merge
        if (len(buffer["text"]) < min_chars and 
            buffer["metadata"]["page"] == chunk["metadata"]["page"] and
            not buffer["metadata"]["is_table"] and 
            not chunk["metadata"]["is_table"]):
            # Merge texts
            buffer["text"] = buffer["text"] + "\n\n" + chunk["text"].split("]", 1)[-1].strip()
            buffer["id"] = buffer["id"]  # Keep first ID
        else:
            merged.append(buffer)
            buffer = chunk
    
    if buffer:
        merged.append(buffer)
    
    return merged


def main():
    print("\n" + "=" * 60)
    print("  PHASE 2: Smart Chunking (Local PDF → chunks)")
    print("=" * 60)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Bucket:  {config.GCS_BUCKET_NAME}")
    
    # Check for local PDF
    pdf_path = config.LOCAL_PDF_PATH
    if not os.path.isfile(pdf_path):
        print(f"\n   ERROR: PDF not found at {pdf_path}")
        print("   Run Phase 1 first or place PDF at LOCAL_PDF_PATH.")
        sys.exit(1)
    
    print(f"\n[Step 1] Parsing local PDF with pdfplumber...")
    print(f"   PDF: {os.path.abspath(pdf_path)}")
    print(f"   Size: {os.path.getsize(pdf_path) / (1024*1024):.2f} MB")
    
    chunks = chunk_pdf(pdf_path, doc_label="doc0")
    print(f"   Raw chunks extracted: {len(chunks)}")
    
    print("\n[Step 2] Merging small chunks...")
    chunks = merge_small_chunks(chunks, min_chars=150)
    print(f"   After merging: {len(chunks)} chunks")
    
    # Stats
    table_chunks = sum(1 for c in chunks if c["metadata"]["is_table"])
    text_chunks = len(chunks) - table_chunks
    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"   Table chunks: {table_chunks}")
    print(f"   Text chunks: {text_chunks}")
    print(f"   Total characters: {total_chars:,}")
    
    if not chunks:
        print("   ERROR: No chunks produced.")
        sys.exit(1)
    
    print("\n[Step 3] Uploading chunks to GCS...")
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = config.GCS_BUCKET_NAME
    chunks_prefix = config.CHUNK_OUTPUT_PREFIX
    
    chunks_jsonl = "\n".join(json.dumps(c) for c in chunks)
    blob_name = f"{chunks_prefix}/chunks.jsonl"
    bucket_obj = storage_client.bucket(bucket)
    out_blob = bucket_obj.blob(blob_name)
    out_blob.upload_from_string(
        chunks_jsonl,
        content_type="application/jsonl",
    )
    uri = f"gs://{bucket}/{blob_name}"
    print(f"   Uploaded: {uri}")
    
    # Show sample chunks
    print("\n[Sample chunks]")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"   {i+1}. [{chunk['metadata']['type']}] {preview}...")
    
    print("\n" + "=" * 60)
    print("  Phase 2 complete! Chunks ready for Phase 3 (embedding + index).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
