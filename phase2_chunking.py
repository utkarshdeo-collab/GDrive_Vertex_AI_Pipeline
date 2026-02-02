"""
Phase 2: Smart Semantic Chunking

Reads Document AI JSON from GCS (Phase 1 output), then:
- Hierarchical context: chunks tagged with section (e.g. [Section: Page N] or inferred heading).
  Section headers include numbered headings (e.g. 3.1, Table 21) so tables get proper context.
- Table preservation: each table as one markdown chunk (no splitting); section header in same chunk.
- Metadata: section_header and is_table for each chunk (for optional reranking).
Output: JSONL of chunks uploaded to gs://<bucket>/chunks/chunks.jsonl
"""
import json
import re
import sys

import config
from google.cloud import documentai_v1 as documentai
from google.cloud import storage


def get_text_from_anchor(doc, text_anchor):
    """Extract text from document using layout text_anchor (start/end indices)."""
    if not text_anchor or not text_anchor.text_segments:
        return ""
    full = doc.text or ""
    parts = []
    for seg in text_anchor.text_segments:
        start = getattr(seg, "start_index", 0) or 0
        end = getattr(seg, "end_index", 0) or 0
        if end > start and start < len(full):
            end = min(end, len(full))
            parts.append(full[start:end])
    return " ".join(parts).strip()


def extract_table_markdown(doc, table, page_num):
    """Extract table as a single markdown string (header + body rows)."""
    rows_md = []
    # Header rows
    for row in getattr(table, "header_rows", []) or []:
        cells = [get_text_from_anchor(doc, c.layout.text_anchor) for c in (getattr(row, "cells", []) or [])]
        rows_md.append(" | ".join(cells))
    # Body rows
    for row in getattr(table, "body_rows", []) or []:
        cells = [get_text_from_anchor(doc, c.layout.text_anchor) for c in (getattr(row, "cells", []) or [])]
        rows_md.append(" | ".join(cells))
    if not rows_md:
        return ""
    # Markdown table: header separator
    header = rows_md[0]
    sep = " | ".join(["---"] * max(1, header.count("|") + 1))
    return "\n".join([header, sep] + rows_md[1:])


def _is_section_header(text):
    """True if text looks like a section header (numbered or 'Table N')."""
    t = (text or "").strip()
    if not t:
        return False
    # e.g. "3.1 Detailed Technology Stack", "Table 21", "4.3 Performance Results"
    if re.match(r"^(\d+\.\d+|\d+\.)\s", t) or re.match(r"^Table\s+\d+", t, re.I):
        return True
    if len(t) < 80 and "." not in t:
        return True
    return False


def chunk_document(doc, doc_label="doc0"):
    """
    Build chunks from a Document AI Document: paragraphs (with section context) and full tables.
    Section headers (e.g. 3.1, Table 21) are detected so tables get the right section in the same chunk.
    Returns list of {"id", "text", "metadata"} with section_header and is_table in metadata.
    """
    chunks = []
    full_text = doc.text or ""
    current_section = "Document"

    pages_list = getattr(doc, "pages", []) or []
    for page_idx, page in enumerate(pages_list):
        page_num = getattr(page, "page_number", None)
        if page_num is None or page_num == 0:
            page_num = page_idx + 1
        section_context = f"Page {page_num}"

        # Paragraphs (or blocks/lines): tag with section; detect section headers for hierarchy
        paras = getattr(page, "paragraphs", []) or getattr(page, "blocks", []) or []
        for i, para in enumerate(paras):
            layout = getattr(para, "layout", None)
            if not layout:
                continue
            text = get_text_from_anchor(doc, getattr(layout, "text_anchor", None))
            if not text or not text.strip():
                continue
            if _is_section_header(text):
                current_section = text.strip()
                section_context = f"Page {page_num} > {current_section}"
            chunk_id = f"{doc_label}_p{page_num}_para{i}"
            chunks.append({
                "id": chunk_id,
                "text": f"[Section: {section_context}]\n{text}",
                "metadata": {
                    "type": "text",
                    "page": page_num,
                    "section_context": section_context,
                    "section_header": section_context,
                    "is_table": False,
                },
            })

        # Tables: one chunk per table (full markdown); section header in same chunk
        for i, table in enumerate(getattr(page, "tables", []) or []):
            table_md = extract_table_markdown(doc, table, page_num)
            if not table_md.strip():
                continue
            chunk_id = f"{doc_label}_p{page_num}_tbl{i}"
            chunks.append({
                "id": chunk_id,
                "text": f"[Section: {section_context}]\n\n{table_md}",
                "metadata": {
                    "type": "table",
                    "page": page_num,
                    "section_context": section_context,
                    "section_header": section_context,
                    "is_table": True,
                },
            })

    return chunks


def list_docai_json_blobs(storage_client, bucket_name, prefix):
    """List .json blobs under GCS prefix (Document AI output)."""
    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    return [b for b in blobs if b.name.endswith(".json")]


def main():
    print("\n" + "=" * 60)
    print("  PHASE 2: Smart Chunking (Document AI → chunks)")
    print("=" * 60)

    bucket = config.GCS_BUCKET_NAME
    docai_prefix = config.GCS_OUTPUT_PREFIX
    chunks_prefix = config.CHUNK_OUTPUT_PREFIX

    storage_client = storage.Client(project=config.PROJECT_ID)

    print("\n[Step 1] Listing Document AI JSON files in GCS...")
    blobs = list_docai_json_blobs(storage_client, bucket, docai_prefix)
    if not blobs:
        print(f"   No JSON files under gs://{bucket}/{docai_prefix}/. Run Phase 1 first.")
        sys.exit(1)
    print(f"   Found {len(blobs)} JSON file(s).")

    print("\n[Step 2] Parsing and chunking...")
    all_chunks = []
    for idx, blob in enumerate(blobs):
        data = blob.download_as_bytes()
        try:
            doc = documentai.Document.from_json(data, ignore_unknown_fields=True)
        except Exception as e:
            print(f"   Skip {blob.name}: {e}")
            continue
        label = f"doc{idx}"
        chunks = chunk_document(doc, doc_label=label)
        all_chunks.extend(chunks)
        print(f"   {blob.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("   No chunks produced.")
        sys.exit(1)
    print(f"   Total chunks: {len(all_chunks)}")

    print("\n[Step 3] Uploading chunks to GCS...")
    chunks_jsonl = "\n".join(json.dumps(c) for c in all_chunks)
    blob_name = f"{chunks_prefix}/chunks.jsonl"
    bucket_obj = storage_client.bucket(bucket)
    out_blob = bucket_obj.blob(blob_name)
    out_blob.upload_from_string(
        chunks_jsonl,
        content_type="application/jsonl",
    )
    uri = f"gs://{bucket}/{blob_name}"
    print(f"   Uploaded: {uri}")

    print("\n" + "=" * 60)
    print("  Phase 2 complete. Chunks ready for Phase 3 (embedding + index).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
