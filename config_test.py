from safetensors import safe_open
from pathlib import Path
import os

# Check if we can inspect GPT-OSS weights locally
test_paths = [
    os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Phi-3-medium-128k-instruct/snapshots/a088b37c71d441ab6d862bb3fcfe6165b3014702"),
]

for path in test_paths:
    if os.path.exists(path):
        safetensors_files = list(Path(path).rglob("*.safetensors"))
        if safetensors_files:
            print(f"Found at: {path}")
            file = safetensors_files[0]
            print(f"Checking: {file}")
            with safe_open(file, "pt", "cpu") as f:
                keys = list(f.keys())
                print(f"\nTotal keys: {len(keys)}")
                print("\nddFirst 80 keys:")
                for key in keys[:80]:
                    print(f"  {key}")
            break
else:
    print("Mistral checkpoint not found in standard locations")
