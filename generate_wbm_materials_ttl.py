import argparse
import csv
import math
from pathlib import Path

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import XSD


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "dataset" / "wbm" / "2023-12-13-wbm-summary.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "ttl_datasets" / "wbm_materials_generated.ttl"

DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")
DATAIND = Namespace("https://k.loria.fr/ontologies/dataset-individuals#")
TCKGMAT = Namespace("https://k.loria.fr/ontologies/tckg_materials#")
TCKGMATI = Namespace("https://k.loria.fr/ontologies/tckg_materials-individuals#")


def slugify(value):
    value = str(value).strip().lower()
    chars = []
    for char in value:
        chars.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(chars).split("_") if part) or "unnamed"


def as_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def as_bool(value):
    if value is None or value == "":
        return None
    lower = str(value).strip().lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    return None


def add_literal(graph, subject, predicate, value, datatype=XSD.string):
    if value is None or value == "":
        return
    graph.add((subject, predicate, Literal(value, datatype=datatype)))


def lit(value, datatype=XSD.string):
    return Literal(value, datatype=datatype).n3()


def uri(value):
    return URIRef(str(value)).n3()


def triple(subject, predicate, obj):
    return f"{subject} {predicate} {obj} .\n"


def material_triples(dataset_uri, material_uri, input_path, row):
    material_id = row["material_id"].strip()
    source = str(input_path.relative_to(BASE_DIR))

    lines = [
        triple(material_uri, "a", "owl:NamedIndividual"),
        triple(material_uri, "a", "tckgmat:Material"),
        triple(material_uri, "rdfs:label", lit(material_id)),
        triple(dataset_uri, "data:containsMaterial", material_uri),
        triple(material_uri, "data:hasMaterialIdentifier", lit(material_id)),
        triple(material_uri, "dcterms:source", lit(source)),
    ]

    text_fields = [
        ("formula", "tckgmat:hasChemicalFormula"),
        ("protostructure_spglib_initial_structure", "tckgmat:hasInitialProtostructure"),
        ("protostructure_spglib", "tckgmat:hasProtostructure"),
    ]
    for column, predicate in text_fields:
        value = row.get(column)
        if value:
            lines.append(triple(material_uri, predicate, lit(value)))

    number_fields = [
        ("n_sites", "tckgmat:hasNumberOfSites"),
        ("volume", "tckgmat:hasVolume"),
    ]
    for column, predicate in number_fields:
        value = as_float(row.get(column))
        if value is not None:
            lines.append(triple(material_uri, predicate, lit(value, XSD.double)))

    unique = as_bool(row.get("unique_prototype"))
    if unique is not None:
        lines.append(triple(material_uri, "tckgmat:isUniquePrototype", lit(unique, XSD.boolean)))

    return lines


def generate(input_path, output_path, limit=None):
    dataset_uri = DATAIND["wbm_test_set"]
    dataset_ref = uri(dataset_uri)
    count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out:
        out.write("@prefix data: <https://k.loria.fr/ontologies/datasetonto#> .\n")
        out.write("@prefix dataind: <https://k.loria.fr/ontologies/dataset-individuals#> .\n")
        out.write("@prefix dcterms: <http://purl.org/dc/terms/> .\n")
        out.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        out.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        out.write("@prefix tckgmat: <https://k.loria.fr/ontologies/tckg_materials#> .\n")
        out.write("@prefix tckgmati: <https://k.loria.fr/ontologies/tckg_materials-individuals#> .\n")
        out.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")

        out.write(triple(uri("https://k.loria.fr/ontologies/tckg_materials-individuals"), "a", "owl:Ontology"))
        out.write(triple(dataset_ref, "a", "owl:NamedIndividual"))
        out.write(triple(dataset_ref, "a", "data:Dataset"))
        out.write(triple(dataset_ref, "rdfs:label", lit("WBM test set")))
        out.write(triple(dataset_ref, "dcterms:source", lit(str(input_path.relative_to(BASE_DIR)))))
        out.write("\n")

        with input_path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                material_id = (row.get("material_id") or "").strip()
                if not material_id:
                    continue

                material_ref = uri(TCKGMATI[slugify(material_id)])
                for line in material_triples(dataset_ref, material_ref, input_path, row):
                    out.write(line)
                out.write("\n")

                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Generate compact WBM material individuals TTL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    count = generate(args.input.resolve(), args.output.resolve(), limit=args.limit)
    print(f"Generated {args.output}")
    print(f"wbm_materials={count}")


if __name__ == "__main__":
    main()
