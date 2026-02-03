"""
Check which GCS regions are allowed by organization policy.
"""
import config
from google.cloud import storage

# Common GCS regions to test
REGIONS_TO_TEST = [
    "us",                # Multi-region US
    "eu",                # Multi-region EU
    "asia",              # Multi-region Asia
    "us-central1",       # Iowa
    "us-east1",          # South Carolina
    "us-east4",          # Northern Virginia
    "us-west1",          # Oregon
    "us-west2",          # Los Angeles
    "us-west3",          # Salt Lake City
    "us-west4",          # Las Vegas
    "us-south1",         # Dallas
    "europe-west1",      # Belgium
    "europe-west2",      # London
    "europe-west3",      # Frankfurt
    "europe-west4",      # Netherlands
    "asia-east1",        # Taiwan
    "asia-southeast1",   # Singapore
    "asia-south1",       # Mumbai
]

def check_region(storage_client, project_id, region):
    """Try to create a bucket in a region, then delete it."""
    import uuid
    test_bucket_name = f"{project_id}-region-test-{uuid.uuid4().hex[:8]}"
    
    try:
        bucket = storage_client.create_bucket(test_bucket_name, location=region)
        # If we got here, the region is allowed - delete the test bucket
        bucket.delete()
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "resourceLocations" in error_msg:
            return False, "Region not allowed by org policy"
        elif "already exists" in error_msg.lower():
            return True, "Region allowed (bucket name conflict)"
        else:
            return False, error_msg[:50]


def main():
    print("\n" + "=" * 60)
    print("  CHECKING ALLOWED GCS REGIONS")
    print(f"  Project: {config.PROJECT_ID}")
    print("=" * 60)
    
    storage_client = storage.Client(project=config.PROJECT_ID)
    
    allowed_regions = []
    
    print("\nTesting regions...\n")
    print(f"{'Region':<20} {'Status':<10} {'Notes'}")
    print("-" * 60)
    
    for region in REGIONS_TO_TEST:
        allowed, error = check_region(storage_client, config.PROJECT_ID, region)
        status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
        notes = error if error else ""
        print(f"{region:<20} {status:<10} {notes}")
        
        if allowed:
            allowed_regions.append(region)
    
    print("\n" + "=" * 60)
    if allowed_regions:
        print(f"  ALLOWED REGIONS: {', '.join(allowed_regions)}")
        print(f"\n  Recommended: Update config.py with:")
        print(f'    LOCATION = "{allowed_regions[0]}"')
    else:
        print("  No allowed regions found. Contact your GCP admin.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
