"""
Validate that required APIs are enabled and accessible for this account/project.

Run when you switch accounts to see which APIs you have access to.
Uses config.py for PROJECT_ID and LOCATION.
"""
import sys

import config
import google.auth
from google.api_core.client_options import ClientOptions
from google.cloud import storage
from google.cloud import documentai_v1 as documentai
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from googleapiclient.discovery import build


def run_check():
    print("\n" + "=" * 60)
    print("  API VALIDATION — Project: %s" % config.PROJECT_ID)
    print("=" * 60)
    print("  Location: %s  |  Doc AI region: %s" % (config.LOCATION, config.DOCAI_LOCATION))
    print("=" * 60)

    try:
        creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as e:
        print("\n  AUTH FAILED: %s" % e)
        print("  Run: gcloud auth application-default login --project=<your-project-id>")
        print("=" * 60 + "\n")
        sys.exit(1)

    results = []

    # 1. Google Drive API
    print("\n[1/5] Google Drive API...")
    try:
        service = build("drive", "v3", credentials=creds)
        service.files().list(pageSize=1).execute()
        print("      OK — enabled and accessible")
        results.append(("Google Drive API", True))
    except Exception as e:
        print("      FAIL — %s" % (str(e)[:80]))
        results.append(("Google Drive API", False))

    # 2. Cloud Storage API
    print("\n[2/5] Cloud Storage API...")
    try:
        client = storage.Client(project=config.PROJECT_ID)
        list(client.list_buckets(max_results=1))
        print("      OK — enabled and accessible")
        results.append(("Cloud Storage API", True))
    except Exception as e:
        print("      FAIL — %s" % (str(e)[:80]))
        results.append(("Cloud Storage API", False))

    # 3. Document AI API
    print("\n[3/5] Document AI API...")
    try:
        opts = ClientOptions(api_endpoint="%s-documentai.googleapis.com" % config.DOCAI_LOCATION)
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        parent = "projects/%s/locations/%s" % (config.PROJECT_ID, config.DOCAI_LOCATION)
        list(client.list_processors(parent=parent))
        print("      OK — enabled and accessible")
        results.append(("Document AI API", True))
    except Exception as e:
        print("      FAIL — %s" % (str(e)[:80]))
        results.append(("Document AI API", False))

    # 4. Vertex AI API
    print("\n[4/5] Vertex AI API...")
    try:
        client = aiplatform_v1.IndexServiceClient(
            client_options={"api_endpoint": "%s-aiplatform.googleapis.com" % config.LOCATION}
        )
        parent = "projects/%s/locations/%s" % (config.PROJECT_ID, config.LOCATION)
        list(client.list_indexes(parent=parent))
        print("      OK — enabled and accessible")
        results.append(("Vertex AI API", True))
    except Exception as e:
        print("      FAIL — %s" % (str(e)[:80]))
        results.append(("Vertex AI API", False))

    # 5. Cloud Resource Manager API
    print("\n[5/5] Cloud Resource Manager API...")
    try:
        service = build("cloudresourcemanager", "v1", credentials=creds)
        service.projects().get(projectId=config.PROJECT_ID).execute()
        print("      OK — enabled and accessible")
        results.append(("Cloud Resource Manager API", True))
    except Exception as e:
        print("      FAIL — %s" % (str(e)[:80]))
        results.append(("Cloud Resource Manager API", False))

    # Summary
    print("\n" + "-" * 60)
    passed = sum(1 for _, ok in results if ok)
    print("  SUMMARY: %d/5 APIs OK" % passed)
    for name, ok in results:
        print("    %s %s" % ("OK" if ok else "FAIL", name))
    print("=" * 60 + "\n")

    if passed < 5:
        sys.exit(1)


if __name__ == "__main__":
    run_check()
