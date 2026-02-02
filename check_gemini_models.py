"""
Check which Gemini models are available in your Vertex AI project.

This script queries the Vertex AI Model Garden to list all available Gemini models
that your project has access to.
"""
import sys

import config
import vertexai
from vertexai import model_garden


def list_available_gemini_models():
    """List all Gemini models available in Model Garden."""
    print("\n" + "=" * 60)
    print("  Checking available Gemini models")
    print("=" * 60)
    print(f"\nProject: {config.PROJECT_ID}")
    print(f"Location: {config.LOCATION}\n")

    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)

    print("Querying Model Garden for Gemini models...\n")

    try:
        # List all models filtered by "gemini"
        all_models = model_garden.list_models(model_filter="gemini")
        
        # Extract just the model names (format: "google/gemini-1.5-flash-002@001" -> "gemini-1.5-flash-002")
        gemini_models = []
        for model_str in all_models:
            if "gemini" in model_str.lower():
                # Format is usually "google/gemini-X@version" or "gemini-X@version"
                # Extract just the model name part
                parts = model_str.split("/")
                if len(parts) > 1:
                    model_part = parts[-1]  # Get last part after "/"
                else:
                    model_part = parts[0]
                # Remove version suffix (@001)
                model_name = model_part.split("@")[0]
                if model_name not in gemini_models:
                    gemini_models.append(model_name)

        if gemini_models:
            # Sort: prefer flash-002, flash-001, then pro, then older versions
            def sort_key(name):
                if "flash-002" in name:
                    return (0, name)
                elif "flash-001" in name:
                    return (1, name)
                elif "flash" in name:
                    return (2, name)
                elif "pro-002" in name:
                    return (3, name)
                elif "pro-001" in name:
                    return (4, name)
                elif "pro" in name:
                    return (5, name)
                else:
                    return (6, name)

            gemini_models.sort(key=sort_key)

            print("AVAILABLE MODELS:")
            print("-" * 60)
            for model_name in gemini_models:
                print(f"  • {model_name}")
            print("\n" + "=" * 60)
            print("RECOMMENDED:")
            print(f"  Use: {gemini_models[0]}")
            print("\nTo use this model, update config.py:")
            print(f'  GEMINI_MODEL = "{gemini_models[0]}"')
            print("\nOr set environment variable:")
            print(f'  set GEMINI_MODEL="{gemini_models[0]}"')
            print("=" * 60 + "\n")
        else:
            print("❌ NO GEMINI MODELS FOUND")
            print("-" * 60)
            print("\nPossible issues:")
            print("  1. Vertex AI API not enabled")
            print("  2. Gemini models not allowlisted for your project")
            print("  3. Incorrect project ID or location")
            print("\nTo fix:")
            print("  1. Enable Vertex AI API in Google Cloud Console")
            print("  2. Go to Vertex AI → Model Garden → Gemini models")
            print("  3. Ensure your project has access to Gemini models")
            print("  4. Check that billing is enabled for your project")
            print("=" * 60 + "\n")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error querying Model Garden: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Vertex AI API is enabled")
        print("  2. Check your project ID and location in config.py")
        print("  3. Verify your authentication credentials")
        print("  4. Try: gcloud auth application-default login")
        sys.exit(1)


if __name__ == "__main__":
    try:
        list_available_gemini_models()
    except Exception as e:
        print(f"\n❌ Error checking models: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Vertex AI API is enabled")
        print("  2. Check your project ID and location in config.py")
        print("  3. Verify your authentication credentials")
        sys.exit(1)
