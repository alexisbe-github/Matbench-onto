import argparse
from pathlib import Path

import requests


GRAPHDB_URL = "http://localhost:7200"
REPOSITORY = "matbench"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MATERIALS_GRAPH = "https://k.loria.fr/graphs/individuals/tckg_materials"

GRAPHS = {
    BASE_DIR / "ontology" / "datasetonto.ttl": "https://k.loria.fr/graphs/ontology/dataset",
    BASE_DIR / "ontology" / "evaluationonto.ttl": "https://k.loria.fr/graphs/ontology/evaluation",
    BASE_DIR / "ontology" / "tckg_materials.ttl": "https://k.loria.fr/graphs/ontology/tckg_materials",
    BASE_DIR / "outputs" / "ttl_datasets" / "matbench_site_context_generated.ttl": "https://k.loria.fr/graphs/individuals/matbench_site_context",
    BASE_DIR / "outputs" / "ttl_datasets" / "materials_properties_generated.ttl": DEFAULT_MATERIALS_GRAPH,
}


def repository_url(graphdb_url, repository):
    return f"{graphdb_url.rstrip('/')}/repositories/{repository}"


def statements_url(graphdb_url, repository):
    return f"{repository_url(graphdb_url, repository)}/statements"


def sparql_update(graphdb_url, repository, update):
    response = requests.post(
        statements_url(graphdb_url, repository),
        data={"update": update},
        timeout=120,
    )
    response.raise_for_status()


def clear_graph(graphdb_url, repository, graph_uri):
    sparql_update(graphdb_url, repository, f"CLEAR GRAPH <{graph_uri}>")


def upload_file(graphdb_url, repository, ttl_path, graph_uri):
    with ttl_path.open("rb") as file:
        response = requests.post(
            statements_url(graphdb_url, repository),
            params={"context": f"<{graph_uri}>"},
            data=file,
            headers={"Content-Type": "text/turtle"},
            timeout=600,
        )
    response.raise_for_status()
    print(f"[OK] {ttl_path.relative_to(BASE_DIR)}")
    print(f"     graph: {graph_uri}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload dataset/material ontology and generated material properties to GraphDB."
    )
    parser.add_argument("--graphdb-url", default=GRAPHDB_URL)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument(
        "--materials-ttl",
        type=Path,
        default=BASE_DIR / "outputs" / "ttl_datasets" / "materials_properties_generated.ttl",
    )
    parser.add_argument("--materials-graph", default=DEFAULT_MATERIALS_GRAPH)
    parser.add_argument("--no-clear", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    graph_map = dict(GRAPHS)
    graph_map.pop(BASE_DIR / "outputs" / "ttl_datasets" / "materials_properties_generated.ttl")
    graph_map[args.materials_ttl.resolve()] = args.materials_graph

    for ttl_path, graph_uri in graph_map.items():
        ttl_path = ttl_path.resolve()
        if not ttl_path.exists():
            raise FileNotFoundError(ttl_path)
        if not args.no_clear:
            clear_graph(args.graphdb_url, args.repository, graph_uri)
            print(f"[CLEAR] {graph_uri}")
        upload_file(args.graphdb_url, args.repository, ttl_path, graph_uri)


if __name__ == "__main__":
    main()
