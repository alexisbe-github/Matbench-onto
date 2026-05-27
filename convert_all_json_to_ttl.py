import os
import sys
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

JSON_DIR = BASE_DIR / "outputs" / "json"
TTL_DIR = BASE_DIR / "outputs" / "ttl"

JSON_TO_TTL_SCRIPT = BASE_DIR / "json_to_ttl.py"


def output_ttl_path(json_path):
    name = json_path.stem
    name = name.replace("_model_extraction", "")
    return TTL_DIR / f"{name}_model_individuals_generated.ttl"


def convert_json_to_ttl(json_path):
    ttl_path = output_ttl_path(json_path)

    env = os.environ.copy()
    env["INPUT_JSON_FILE"] = str(json_path)
    env["OUTPUT_TTL_FILE"] = str(ttl_path)

    print(f"\n=== CONVERT ===")
    print(f"JSON: {json_path}")
    print(f"TTL:  {ttl_path}")

    subprocess.run(
        [sys.executable, str(JSON_TO_TTL_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        check=True,
    )

    print(f"[OK] {ttl_path}")



TTL_DIR.mkdir(parents=True, exist_ok=True)

json_files = sorted(JSON_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files.")

for json_path in json_files:
    try:
        convert_json_to_ttl(json_path)
    except Exception as error:
        print(f"[ERROR] {json_path}")
        print(error)


