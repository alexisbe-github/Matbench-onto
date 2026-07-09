import os
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Change only this value when you want to try another OpenRouter model.
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

MODEL_SLUG = "tece_oam_rra_1_0"
YAML_FILE = BASE_DIR / "model_yamls" / f"{MODEL_SLUG}.yml"
PDF_FILE = BASE_DIR / "papers" / f"{MODEL_SLUG}.pdf"
OUTPUT_JSON_FILE = BASE_DIR / "outputs" / "json" / f"{MODEL_SLUG}_model_extraction.json"
OUTPUT_TTL_FILE = BASE_DIR / "outputs" / "ttl" / f"{MODEL_SLUG}_model_individuals_generated.ttl"
REPAIRED_TTL_FILE = BASE_DIR / "outputs" / "ttl_repaired" / OUTPUT_TTL_FILE.name

PDF_URL = "https://arxiv.org/pdf/2509.14961.pdf"
MODEL_PAGE_URL = "https://matbench-discovery.materialsproject.org/models/tece-oam-rra-1.0"


def run(command, env=None):
    print(f"\n> {' '.join(command)}")
    subprocess.run(command, cwd=BASE_DIR, env=env, check=True)


def main():
    env = os.environ.copy()
    env["OPENROUTER_MODEL"] = OPENROUTER_MODEL
    env["YAML_FILE"] = str(YAML_FILE)
    env["PDF_FILE"] = str(PDF_FILE)
    env["PDF_URL"] = PDF_URL
    env["MODEL_PAGE_URL"] = MODEL_PAGE_URL
    env["OUTPUT_JSON_FILE"] = str(OUTPUT_JSON_FILE)

    run([sys.executable, "seed_kg_open_router.py"], env=env)

    env["INPUT_JSON_FILE"] = str(OUTPUT_JSON_FILE)
    env["OUTPUT_TTL_FILE"] = str(OUTPUT_TTL_FILE)
    run([sys.executable, "json_to_ttl.py"], env=env)

    REPAIRED_TTL_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_TTL_FILE, REPAIRED_TTL_FILE)

    run([sys.executable, "validate_shacl.py", "--ttl", str(REPAIRED_TTL_FILE)])


if __name__ == "__main__":
    main()
