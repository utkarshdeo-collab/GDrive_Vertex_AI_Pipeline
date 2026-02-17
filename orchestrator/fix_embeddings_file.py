"""
Fix existing embeddings.json file by removing sparse_embedding: null fields.

Run this script to fix an existing embeddings.json file that has sparse_embedding: null.
"""
import json
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def fix_embeddings_file(input_path: str, output_path: str = None):
    """Fix embeddings file by removing sparse_embedding: null and numeric_restricts if empty."""
    if output_path is None:
        output_path = input_path
    
    print(f"Reading embeddings from: {input_path}")
    fixed_lines = []
    fixed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                
                # Remove sparse_embedding if it's null or missing
                if "sparse_embedding" in entry:
                    if entry["sparse_embedding"] is None:
                        del entry["sparse_embedding"]
                        fixed_count += 1
                
                # Remove numeric_restricts if it's empty (optional cleanup)
                if "numeric_restricts" in entry and entry["numeric_restricts"] == []:
                    del entry["numeric_restricts"]
                
                fixed_lines.append(json.dumps(entry))
                
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Fixed {fixed_count} entries with sparse_embedding: null")
    print(f"Writing fixed embeddings to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(fixed_lines))
    
    print(f"✓ Fixed file saved: {output_path}")
    print(f"  Total entries: {len(fixed_lines)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix embeddings.json file by removing sparse_embedding: null")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input embeddings file path (default: temp_embeddings/embeddings.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output embeddings file path (default: overwrites input)"
    )
    
    args = parser.parse_args()
    
    # Default path
    if args.input is None:
        input_path = Path(__file__).parent.parent / "temp_embeddings" / "embeddings.json"
    else:
        input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    output_path = args.output if args.output else input_path
    
    fix_embeddings_file(str(input_path), str(output_path))
