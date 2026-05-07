from pathlib import Path
import json

from rdflib import Graph
from rdflib.namespace import RDF, OWL


ONTOLOGY_DIR = "ontology"
OUTPUT_FILE = "ontology_context.json"


def short_name(uri):
    text = str(uri)
    if "#" in text:
        return text.split("#")[-1]
    return text.rstrip("/").split("/")[-1]


def load_graph(folder):
    graph = Graph()

    for ttl_file in Path(folder).glob("*.ttl"):
        print(f"Loading {ttl_file}")
        graph.parse(ttl_file, format="turtle")

    return graph


def extract_context(graph):
    def collect(rdf_type):
        items = []

        for uri in graph.subjects(RDF.type, rdf_type):
            items.append({
                "iri": str(uri),
                "name": short_name(uri),
            })

        return sorted(items, key=lambda x: x["name"])

    return {
        "classes": collect(OWL.Class),
        "object_properties": collect(OWL.ObjectProperty),
        "datatype_properties": collect(OWL.DatatypeProperty),
    }


graph = load_graph(ONTOLOGY_DIR)
context = extract_context(graph)

Path(OUTPUT_FILE).write_text(
    json.dumps(context, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Written {OUTPUT_FILE}")
print(f"Classes: {len(context['classes'])}")
print(f"Object properties: {len(context['object_properties'])}")
print(f"Datatype properties: {len(context['datatype_properties'])}")