from pathlib import Path
import requests

GRAPHDB_URL = "http://localhost:7200"
REPOSITORY = "matbench"

TTL_DIR = Path("outputs/ttl")

GRAPH_BASE = "https://k.loria.fr/graphs/individuals"


def graph_name_from_ttl(ttl_path):
    name = ttl_path.stem

    name = name.replace("_model_individuals_generated", "")
    name = name.lower()

    return f"{GRAPH_BASE}/{name}"


def upload_ttl_to_named_graph(ttl_path):
    graph_uri = graph_name_from_ttl(ttl_path)

    url = (
        f"{GRAPHDB_URL}/repositories/{REPOSITORY}/statements"
        f"?context=<{graph_uri}>"
    )

    with open(ttl_path, "rb") as file:
        response = requests.post(
            url,
            data=file,
            headers={
                "Content-Type": "text/turtle"
            }
        )

    response.raise_for_status()

    print(f"[OK] {ttl_path.name}")
    print(f"     graph: {graph_uri}")



ttl_files = sorted(TTL_DIR.glob("*.ttl"))

if not ttl_files:
    print(f"No TTL files found in {TTL_DIR}")


for ttl_path in ttl_files:
    try:
        upload_ttl_to_named_graph(ttl_path)
    except Exception as error:
        print(f"[ERROR] {ttl_path}")
        print(error)

