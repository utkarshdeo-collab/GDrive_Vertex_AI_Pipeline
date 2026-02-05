"""
Phase 2: Smart Chunking from Document AI output (Gemini flow).

- Layout-based chunking: break by Document AI paragraphs and tables (no fixed-size splits).
- 10–15% overlap between consecutive chunks so context isn't lost at borders.
- Metadata: file_name, page_number, section_title, type (text/table).

Input:  Document AI JSON in gs://{GCS_PDF_INPUT_BUCKET}/{DOCAI_OUTPUT_PREFIX}/<doc_stem>/
Output: gs://{CHUNKS_BUCKET}/{CHUNK_OUTPUT_PREFIX}/chunks.jsonl

  python phase2_chunking.py
"""
import json
import os
import re
import sys

import config
from google.cloud import storage


def _get_text_from_anchor(doc_text: str, elem: dict) -> str:
    """Extract text for an element using text_anchor (camelCase or snake_case).
    
    Document AI elements may have textAnchor at the top level or nested in layout.
    E.g., paragraph.layout.textAnchor or table.layout.textAnchor
    """
    # Try direct textAnchor first
    anchor = elem.get("textAnchor") or elem.get("text_anchor")
    
    # If not found, try layout.textAnchor (common for paragraphs, blocks, tables)
    if not anchor:
        layout = elem.get("layout") or elem.get("Layout") or {}
        anchor = layout.get("textAnchor") or layout.get("text_anchor")
    
    if not anchor:
        return ""
    
    segments = anchor.get("textSegments") or anchor.get("text_segments") or []
    if not doc_text or not segments:
        return ""
    
    parts = []
    for seg in segments:
        start = int(seg.get("startIndex", seg.get("start_index", 0)))
        end = int(seg.get("endIndex", seg.get("end_index", 0)))
        if start < end:
            parts.append(doc_text[start:end])
    return "".join(parts).strip()


def _is_likely_header(text: str) -> bool:
    """Heuristic: short, no trailing period, title-like."""
    if not text or len(text) > 120:
        return False
    t = text.strip()
    if not t or t.endswith(".") or t.endswith(":"):
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+\w", t):
        return True
    if t.isupper() and len(t) < 80:
        return True
    words = t.split()
    if len(words) <= 8 and words and words[0][:1].isupper():
        return True
    return False


def _table_to_markdown(table: dict, doc_text: str) -> str:
    """Build markdown from Document AI table (body_rows/header_rows with cells)."""
    body_rows = table.get("bodyRows") or table.get("body_rows") or []
    header_rows = table.get("headerRows") or table.get("header_rows") or []
    all_rows = list(header_rows) + list(body_rows)
    if not all_rows:
        return _get_text_from_anchor(doc_text, table)

    rows_text = []
    for row in all_rows:
        cells = row.get("cells") or []
        cell_texts = []
        for cell in cells:
            ct = _get_text_from_anchor(doc_text, cell)
            cell_texts.append(ct.replace("|", "\\|").replace("\n", " "))
        rows_text.append("| " + " | ".join(cell_texts) + " |")
    if not rows_text:
        return _get_text_from_anchor(doc_text, table)
    sep = "| " + " | ".join(["---"] * max(1, len(rows_text[0].split("|")) - 2)) + " |"
    return rows_text[0] + "\n" + sep + "\n" + "\n".join(rows_text[1:])


def _unwrap_document(raw: dict) -> dict:
    """Handle wrapped batch output: e.g. {"document": {...}} or root is the document."""
    if "document" in raw:
        return raw["document"]
    if "results" in raw and raw["results"]:
        first = raw["results"][0]
        if isinstance(first, dict) and "document" in first:
            return first["document"]
    return raw


def _get_pages(doc: dict) -> list:
    """Get pages array (camelCase or snake_case)."""
    return doc.get("pages") or doc.get("Pages") or []


def _collect_elements(doc: dict, file_name: str) -> list:
    """Return list of (page_num, type, text, section_hint) in reading order.
    Supports Document OCR (lines) and Layout Parser (paragraphs, blocks, tables)."""
    doc = _unwrap_document(doc)
    full_text = doc.get("text") or ""
    pages = _get_pages(doc)
    elements = []
    current_section = "Document"

    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1
        # Layout Parser: paragraphs first
        for para in page.get("paragraphs") or page.get("Paragraphs") or []:
            text = _get_text_from_anchor(full_text, para)
            if not text or len(text) < 3:
                continue
            if _is_likely_header(text):
                current_section = text[:100]
            elements.append((page_num, "text", text, current_section))

        # Blocks
        for block in page.get("blocks") or page.get("Blocks") or []:
            text = _get_text_from_anchor(full_text, block)
            if not text or len(text) < 3:
                continue
            if _is_likely_header(text):
                current_section = text[:100]
            elements.append((page_num, "text", text, current_section))

        # Tables
        for table in page.get("tables") or page.get("Tables") or []:
            text = _table_to_markdown(table, full_text)
            if not text or len(text) < 5:
                text = _get_text_from_anchor(full_text, table)
            if text:
                elements.append((page_num, "table", text, current_section))

    # Document OCR fallback: if no paragraphs/blocks/tables, use lines per page
    if not elements and full_text.strip():
        for page_idx, page in enumerate(pages):
            page_num = page_idx + 1
            lines = page.get("lines") or page.get("Lines") or []
            line_texts = []
            for line in lines:
                t = _get_text_from_anchor(full_text, line)
                if t:
                    line_texts.append(t)
            if line_texts:
                buf, buf_len = [], 0
                for t in line_texts:
                    buf.append(t)
                    buf_len += len(t)
                    if buf_len >= 200 or len(buf) >= 5:
                        combined = "\n".join(buf)
                        if _is_likely_header(combined):
                            current_section = combined[:100]
                        elements.append((page_num, "text", combined, current_section))
                        buf, buf_len = [], 0
                if buf:
                    combined = "\n".join(buf)
                    if len(combined) >= 10:
                        elements.append((page_num, "text", combined, current_section))

    # Last resort: chunk full text by double newline
    if not elements and full_text.strip():
        parts = re.split(r"\n\s*\n", full_text.strip())
        for p in parts:
            p = p.strip()
            if len(p) >= 10:
                elements.append((1, "text", p, "Document"))

    return elements


def _elements_to_chunks(elements: list, file_name: str, doc_label: str, min_text_chars: int = 80) -> list:
    """Convert elements to chunks; merge very small consecutive text elements."""
    chunks = []
    for i, (page_num, typ, text, section) in enumerate(elements):
        chunk_id = f"{doc_label}_p{page_num}_{typ}_{i}"
        chunks.append({
            "id": chunk_id,
            "text": f"[Page {page_num} | Section: {section}]\n\n{text}",
            "metadata": {
                "file_name": file_name,
                "page_number": page_num,
                "section_title": section,
                "type": typ,
                "is_table": typ == "table",
            },
        })
    return chunks


def _apply_overlap(chunks: list, overlap_ratio: float) -> list:
    """Prefix each chunk (except first) with last overlap_ratio of previous chunk."""
    if overlap_ratio <= 0 or len(chunks) <= 1:
        return chunks
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
            continue
        prev_text = out[-1]["text"]
        # Strip [Page ...] prefix for overlap segment
        prev_body = prev_text.split("\n\n", 1)[-1] if "\n\n" in prev_text else prev_text
        overlap_len = max(50, int(len(prev_body) * overlap_ratio))
        overlap_start = len(prev_body) - overlap_len
        if overlap_start > 0:
            overlap_snippet = "... " + prev_body[overlap_start:].strip()
            new_text = overlap_snippet + "\n\n" + c["text"]
        else:
            new_text = c["text"]
        out.append({**c, "text": new_text})
    return out


def get_doc_ai_blobs(storage_client, bucket_name: str, folder_prefix: str) -> list:
    """List blob names under folder_prefix that end with .json."""
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_prefix))
    return [b.name for b in blobs if b.name.endswith(".json")]


def main():
    print("\n" + "=" * 60)
    print("  PHASE 2: Chunking (Document AI → layout-based chunks)")
    print("=" * 60)
    print("\n  Layout-based chunking (no fixed-size). 10–15% overlap.")
    print("=" * 60)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Doc AI output: gs://{config.GCS_PDF_INPUT_BUCKET}/{config.DOCAI_OUTPUT_PREFIX}/")
    print(f"  Chunks output: gs://{config.CHUNKS_BUCKET}/{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl")

    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket_name = config.GCS_PDF_INPUT_BUCKET
    out_bucket = config.CHUNKS_BUCKET
    docai_prefix = config.DOCAI_OUTPUT_PREFIX.strip().rstrip("/") + "/"
    overlap = getattr(config, "CHUNK_OVERLAP_RATIO", 0.12)

    print("\n[Step 1] Listing Document AI output folders...")
    # List blobs under document-ai-output/, group by first path segment (doc_stem)
    all_blobs = list(storage_client.list_blobs(bucket_name, prefix=docai_prefix))
    stems_seen = set()
    for b in all_blobs:
        parts = b.name[len(docai_prefix):].split("/")
        if parts and parts[0]:
            stems_seen.add(parts[0])
    doc_folders = sorted(stems_seen)
    if not doc_folders:
        print(f"   No Document AI output under gs://{bucket_name}/{docai_prefix}")
        print("   Run phase1b_parsing.py first.")
        sys.exit(1)
    print(f"   Found {len(doc_folders)} document(s)")

    print("\n[Step 2] Loading JSON and building layout-based chunks...")
    all_chunks = []
    for stem in doc_folders:
        folder_prefix = docai_prefix + stem + "/"
        json_blobs = [n for n in get_doc_ai_blobs(storage_client, bucket_name, folder_prefix)]
        if not json_blobs:
            print(f"   Skip {stem}: no JSON")
            continue
        # Prefer document.json (sync output), else first .json
        preferred = [n for n in json_blobs if n.endswith("document.json")]
        json_blob_name = preferred[0] if preferred else json_blobs[0]
        blob = storage_client.bucket(bucket_name).blob(json_blob_name)
        try:
            doc = json.loads(blob.download_as_text())
        except Exception as e:
            print(f"   Skip {stem}: failed to load JSON — {e}")
            continue
        
        file_name = stem.replace("_", " ") + ".pdf"
        elements = _collect_elements(doc, file_name)
        if not elements:
            print(f"   Skip {stem}: no elements extracted")
            continue
        doc_chunks = _elements_to_chunks(elements, file_name, stem)
        doc_chunks = _apply_overlap(doc_chunks, overlap)
        all_chunks.extend(doc_chunks)
        print(f"   {stem}: {len(elements)} elements → {len(doc_chunks)} chunks")

    if not all_chunks:
        print("   ERROR: No chunks produced.")
        sys.exit(1)

    print(f"\n   Total chunks: {len(all_chunks)}")
    print(f"   Overlap: {int(overlap * 100)}%")

    print("\n[Step 3] Uploading chunks to GCS...")
    chunks_jsonl = "\n".join(json.dumps(c) for c in all_chunks)
    blob_name = f"{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl"
    out_blob = storage_client.bucket(out_bucket).blob(blob_name)
    out_blob.upload_from_string(chunks_jsonl, content_type="application/jsonl")
    uri = f"gs://{out_bucket}/{blob_name}"
    print(f"   Uploaded: {uri}")

    print("\n[Sample chunks]")
    for i, c in enumerate(all_chunks[:3]):
        preview = c["text"][:180].replace("\n", " ")
        print(f"   {i+1}. [{c['metadata']['type']}] {preview}...")

    print("\n" + "=" * 60)
    print("  Phase 2 complete. Chunks ready for Phase 3 (embedding + index).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
