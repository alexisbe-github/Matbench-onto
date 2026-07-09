import argparse
from pathlib import Path

import requests


GRAPHDB_URL = "http://localhost:7200"
REPOSITORY = "matbench"

TTL_DIR = Path("outputs/ttl_repaired")

INDIVIDUALS_GRAPH_BASE = "https://k.loria.fr/graphs/individuals"

ONTOLOGY_GRAPHS_TO_KEEP = {
    "https://k.loria.fr/graphs/ontology/architecture",
    "https://k.loria.fr/graphs/ontology/dataset",
    "https://k.loria.fr/graphs/ontology/training",
}


def repository_url(graphdb_url, repository):
    return f"{graphdb_url.rstrip('/')}/repositories/{repository}"


def statements_url(graphdb_url, repository):
    return f"{repository_url(graphdb_url, repository)}/statements"


def graph_name_from_ttl(ttl_path):
    name = ttl_path.stem
    name = name.replace("_model_individuals_generated", "")
    name = name.lower()

    return f"{INDIVIDUALS_GRAPH_BASE}/{name}"


def sparql_query(graphdb_url, repository, query):
    response = requests.get(
        repository_url(graphdb_url, repository),
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def sparql_update(graphdb_url, repository, update):
    response = requests.post(
        statements_url(graphdb_url, repository),
        data={"update": update},
        timeout=60,
    )
    response.raise_for_status()


def list_named_graphs(graphdb_url, repository):
    result = sparql_query(
        graphdb_url,
        repository,
        "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } } ORDER BY ?g",
    )

    graphs = []

    for binding in result["results"]["bindings"]:
        graphs.append(binding["g"]["value"])

    return graphs


def clear_graph(graphdb_url, repository, graph_uri):
    sparql_update(
        graphdb_url,
        repository,
        f"CLEAR GRAPH <{graph_uri}>",
    )


def clear_non_ontology_graphs(graphdb_url, repository):
    existing_graphs = list_named_graphs(graphdb_url, repository)
    deleted_graphs = []

    for graph_uri in existing_graphs:
        if graph_uri in ONTOLOGY_GRAPHS_TO_KEEP:
            continue

        clear_graph(graphdb_url, repository, graph_uri)
        deleted_graphs.append(graph_uri)

    return deleted_graphs


def upload_ttl_to_named_graph(graphdb_url, repository, ttl_path):
    graph_uri = graph_name_from_ttl(ttl_path)

    with open(ttl_path, "rb") as file:
        response = requests.post(
            statements_url(graphdb_url, repository),
            params={"context": f"<{graph_uri}>"},
            data=file,
            headers={"Content-Type": "text/turtle"},
            timeout=120,
        )

    response.raise_for_status()

    print(f"[OK] {ttl_path.name}")
    print(f"     graph: {graph_uri}")


def ttl_files_from_dir(ttl_dir):
    ttl_dir = Path(ttl_dir)

    return sorted(
        path for path in ttl_dir.glob("*.ttl")
        if not path.name.startswith("__")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace Matbench individual graphs in GraphDB while keeping "
            "the ontology named graphs."
        )
    )
    parser.add_argument("--graphdb-url", default=GRAPHDB_URL)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--ttl-dir", default=str(TTL_DIR))
    parser.add_argument(
        "--ttl-file",
        type=Path,
        help="Upload one TTL file instead of every TTL file in --ttl-dir.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not delete existing non-ontology named graphs before upload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ttl_file:
        ttl_files = [args.ttl_file]
    else:
        ttl_files = ttl_files_from_dir(args.ttl_dir)

    if not ttl_files:
        print(f"No TTL files found in {args.ttl_dir}")
        return

    if args.ttl_file:
        graph_uri = graph_name_from_ttl(args.ttl_file)
        clear_graph(args.graphdb_url, args.repository, graph_uri)
        print(f"[CLEAR] {graph_uri}")
    elif args.no_clear:
        print("[SKIP] Existing graphs were not cleared.")
    else:
        deleted_graphs = clear_non_ontology_graphs(
            args.graphdb_url,
            args.repository,
        )

        print(f"[CLEAR] Deleted {len(deleted_graphs)} non-ontology graph(s).")

        for graph_uri in ONTOLOGY_GRAPHS_TO_KEEP:
            print(f"[KEEP] {graph_uri}")

    failed = []

    for ttl_path in ttl_files:
        try:
            upload_ttl_to_named_graph(
                args.graphdb_url,
                args.repository,
                ttl_path,
            )
        except Exception as error:
            print(f"[ERROR] {ttl_path}")
            print(error)
            failed.append(ttl_path)

    if failed:
        print(f"\nUpload failed for {len(failed)} file(s).")

    print(f"\nUploaded {len(ttl_files) - len(failed)} / {len(ttl_files)} file(s).")


if __name__ == "__main__":
    main()
