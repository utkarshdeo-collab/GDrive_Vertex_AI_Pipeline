"""
Phase 4: GCS Record Splitter
============================

This module is responsible for splitting the monolithic `nexus_vectors_input.jsonl` 
file into individual JSON objects stored in Google Cloud Storage (GCS).

This step is critical for the "Serverless RAG" architecture, enabling Agents to 
perform O(1) lookups of the full record content by ID, bypassing the need for 
a dedicated database or in-memory loading of the entire dataset.

Classes:
    GCSRecordSplitter: Handles the parsing and parallel upload of records.

Usage:
    Run directly to execute the splitting process:
    $ python phase4_split_records_to_gcs.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List

from google.cloud import storage
from tqdm import tqdm

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    import nexus_config as config


class GCSRecordSplitter:
    """
    Manages the splitting and uploading of JSONL records to GCS.

    Attributes:
        project_id (str): GCP Project ID.
        bucket_name (str): Target GCS Bucket Name.
        source_file (Path): Path to the source JSONL file.
        destination_prefix (str): Folder prefix in the bucket for the records.
        max_workers (int): Number of parallel upload threads.
    """

    def __init__(self, 
                 project_id: str, 
                 bucket_name: str, 
                 source_file: Path, 
                 destination_prefix: str = "records",
                 max_workers: int = 50):
        """
        Initialize the splitter configurations.

        Args:
            project_id (str): GCP Project ID.
            bucket_name (str): Target GCS Bucket Name.
            source_file (Path): Path to the source JSONL file.
            destination_prefix (str, optional): Folder prefix. Defaults to "records".
            max_workers (int, optional): Thread count. Defaults to 50.
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.source_file = source_file
        self.destination_prefix = destination_prefix
        self.max_workers = max_workers
        
        self.storage_client = storage.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def _upload_single_record(self, record: Dict[str, Any]) -> bool:
        """
        Uploads a single record to GCS.

        Args:
            record (Dict[str, Any]): The record dictionary. Must contain an 'id' key.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            record_id = record.get("id")
            if not record_id:
                return False

            blob_name = f"{self.destination_prefix}/{record_id}.json"
            blob = self.bucket.blob(blob_name)
            
            blob.upload_from_string(
                data=json.dumps(record, indent=2),
                content_type="application/json"
            )
            return True
        except Exception as e:
            print(f"[Error] Failed to upload {record.get('id', 'unknown')}: {e}")
            return False

    def process(self) -> None:
        """
        Executes the split and upload process.
        """
        print("=" * 80)
        print("PHASE 4: Split Records to GCS (Serverless Lookup)")
        print("=" * 80)
        print(f"Source File: {self.source_file}")
        print(f"Target: gs://{self.bucket_name}/{self.destination_prefix}/")
        
        if not self.source_file.exists():
            print(f"[Error] Input file not found: {self.source_file}")
            print(f"Please run Phase 1 first or check path.")
            sys.exit(1)

        print("[Step 1] Reading source file...")
        records = []
        with open(self.source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        total_records = len(records)
        print(f"[Info] Found {total_records} records to process.")
        
        print(f"[Step 2] Uploading to GCS (Max Workers: {self.max_workers})...")
        start_time = time.time()
        
        success_count = 0
        failure_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._upload_single_record, record): record['id'] 
                for record in records
            }
            
            # Monitor progress
            for future in tqdm(as_completed(future_to_id), total=total_records, unit="file"):
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1

        duration = time.time() - start_time
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total Uploaded: {success_count}")
        print(f"Failed:         {failure_count}")
        print(f"Time Taken:     {duration:.2f} seconds")
        print(f"Rate:           {success_count / duration:.1f} files/sec")
        print(f"Location:       gs://{self.bucket_name}/{self.destination_prefix}/")
        print("=" * 80)


if __name__ == "__main__":
    # Configuration
    # CORRECT PATH: Input JSONL is in the same directory as this script (Nexus_Hybrid_Engine)
    # We use Path(__file__).parent to be robust regardless of where the command is run from.
    INPUT_FILE = Path(__file__).resolve().parent / "nexus_vectors_input.jsonl"
    
    # Use a specific folder for these lookup records
    TARGET_PREFIX = "lookup_records"

    splitter = GCSRecordSplitter(
        project_id=config.PROJECT_ID,
        bucket_name=config.GCS_BUCKET_NAME,
        source_file=INPUT_FILE,
        destination_prefix=TARGET_PREFIX,
        max_workers=50  # Adjust based on bandwidth
    )
    
    splitter.process()
