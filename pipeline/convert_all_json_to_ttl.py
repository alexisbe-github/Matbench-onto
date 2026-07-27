import argparse
import os
import sys
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

BACKENDS_DIR = BASE_DIR / "pipeline" / "backends"

PROFILES = {
    "longcat": {
        "json_dir": BASE_DIR / "outputs" / "json",
        "ttl_dir": BASE_DIR / "outputs" / "ttl",
        "converter": BACKENDS_DIR / "longcat" / "json_to_ttl.py",
    },
    "free": {
        "json_dir": BASE_DIR / "outputs" / "free_llm" / "json",
        "ttl_dir": BASE_DIR / "outputs" / "free_llm" / "ttl",
        "converter": BACKENDS_DIR / "free" / "json_to_ttl.py",
    },
}


def output_ttl_path(json_path, ttl_dir):
    name = json_path.stem
    name = name.replace("_model_extraction", "")
    return ttl_dir / f"{name}_model_individuals_generated.ttl"


def convert_json_to_ttl(json_path, ttl_dir, converter):
    ttl_path = output_ttl_path(json_path, ttl_dir)

    env = os.environ.copy()
    env["INPUT_JSON_FILE"] = str(json_path)
    env["OUTPUT_TTL_FILE"] = str(ttl_path)

    print(f"\n=== CONVERT ===")
    print(f"JSON: {json_path}")
    print(f"TTL:  {ttl_path}")

    subprocess.run(
        [sys.executable, str(converter)],
        cwd=str(BASE_DIR),
        env=env,
        check=True,
    )

    print(f"[OK] {ttl_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert every extracted model JSON file for one LLM profile."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="longcat",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PROFILES[args.profile]
    json_dir = config["json_dir"]
    ttl_dir = config["ttl_dir"]
    converter = config["converter"]

    ttl_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(json_dir.glob("*.json"))

    print(f"Profile: {args.profile}")
    print(f"Found {len(json_files)} JSON files.")

    for json_path in json_files:
        try:
            convert_json_to_ttl(json_path, ttl_dir, converter)
        except Exception as error:
            print(f"[ERROR] {json_path}")
            print(error)


if __name__ == "__main__":
    main()


