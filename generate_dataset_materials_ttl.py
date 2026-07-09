import argparse
import csv
import gzip
import math
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, OWL, XSD


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "dataset"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "ttl_datasets" / "materials_properties_generated.ttl"

DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")
DATAIND = Namespace("https://k.loria.fr/ontologies/dataset-individuals#")
ARCHIND = Namespace("https://k.loria.fr/ontologies/architectureonto-individuals#")
TCKGMAT = Namespace("https://k.loria.fr/ontologies/tckg_materials#")
TCKGMATI = Namespace("https://k.loria.fr/ontologies/tckg_materials-individuals#")

ID_COLUMNS = {
    "",
    "id",
    "index",
    "material_id",
    "mp_id",
    "material",
    "structure",
    "struct_col",
}

PROPERTY_UNITS = {
    "e_form_per_atom": "eV/atom",
    "energy_pa": "eV/atom",
    "volume_pa": "angstrom^3/atom",
    "entropy": "J/mol/K",
    "heat_capacity": "J/mol/K",
    "free_energy": "eV",
    "max_freq": "cm^-1",
    "avg_freq": "cm^-1",
}


def slugify(value):
    value = str(value).strip().lower()
    chars = []
    for char in value:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "unnamed"


def csv_files(dataset_dir):
    pred_root = dataset_dir / "pred_files"
    for path in sorted(pred_root.rglob("*.csv")):
        yield path
    phonondb = dataset_dir / "phonondb" / "pbesol.csv"
    if phonondb.exists():
        yield phonondb


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return path.open("r", encoding="utf-8", errors="ignore", newline="")


def material_id_from_row(row):
    for key in ("material_id", "mp_id"):
        value = row.get(key)
        if value:
            return value.strip()
    return None


def dataset_uri_for_file(path):
    normalized = path.as_posix().lower()
    if "/phonondb/" in normalized:
        return DATAIND["phonondb_pbesol"]
    return DATAIND["wbm_test_set"]


def model_slug_for_file(path):
    parts = path.parts
    if "pred_files" not in parts:
        return None
    index = parts.index("pred_files")
    if index + 1 >= len(parts):
        return None
    return slugify(parts[index + 1])


def is_number(value):
    if value is None or value == "":
        return False
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number)


def literal_from_string(value):
    if value in {"True", "False", "true", "false"}:
        return Literal(value.lower() == "true", datatype=XSD.boolean)
    if is_number(value):
        number = float(value)
        if number.is_integer() and "." not in str(value) and "e" not in str(value).lower():
            return Literal(int(number), datatype=XSD.integer)
        return Literal(number, datatype=XSD.double)
    return Literal(value)


def unit_for_property(property_name):
    lower = property_name.lower()
    for key, unit in PROPERTY_UNITS.items():
        if key in lower:
            return unit
    return None


def property_columns(fieldnames):
    columns = []
    for field in fieldnames or []:
        if field is None:
            continue
        name = field.strip()
        if not name or name.lower() in ID_COLUMNS:
            continue
        columns.append(name)
    return columns


def add_named(graph, uri, rdf_class, label):
    graph.add((uri, RDF.type, OWL.NamedIndividual))
    graph.add((uri, RDF.type, rdf_class))
    graph.add((uri, RDFS.label, Literal(str(label), datatype=XSD.string)))


def add_dataset_basics(graph, dataset_uri, label):
    add_named(graph, dataset_uri, DATA.Dataset, label)


def add_property_attribute(graph, property_name):
    attribute_uri = DATAIND[f"{slugify(property_name)}_attribute"]
    add_named(graph, attribute_uri, DATA.DatasetAttribute, property_name)
    return attribute_uri


def generate(dataset_dir, output_path, limit_per_file=None, files_limit=None):
    graph = Graph()
    graph.bind("data", DATA)
    graph.bind("dataind", DATAIND)
    graph.bind("archind", ARCHIND)
    graph.bind("tckgmat", TCKGMAT)
    graph.bind("tckgmati", TCKGMATI)
    graph.bind("dcterms", DCTERMS)
    graph.bind("owl", OWL)
    graph.bind("xsd", XSD)
    graph.bind("rdfs", RDFS)

    graph.add((URIRef("https://k.loria.fr/ontologies/tckg_materials-individuals"), RDF.type, OWL.Ontology))

    files = list(csv_files(dataset_dir))
    if files_limit is not None:
        files = files[:files_limit]

    material_count = 0
    property_count = 0
    seen_materials = set()

    for path in files:
        dataset_uri = dataset_uri_for_file(path)
        dataset_label = "PhononDB PBEsol" if "phonondb" in path.parts else "WBM test set"
        add_dataset_basics(graph, dataset_uri, dataset_label)
        model_slug = model_slug_for_file(path)
        model_variant_uri = ARCHIND[f"{model_slug}_variant"] if model_slug else None

        with open_text(path) as file:
            reader = csv.DictReader(file)
            columns = property_columns(reader.fieldnames)
            rows_seen = 0
            for row in reader:
                material_id = material_id_from_row(row)
                if not material_id:
                    continue
                material_slug = slugify(material_id)
                material_uri = TCKGMATI[material_slug]

                if material_uri not in seen_materials:
                    add_named(graph, material_uri, TCKGMAT.Material, material_id)
                    graph.add((material_uri, DATA.hasMaterialIdentifier, Literal(material_id, datatype=XSD.string)))
                    seen_materials.add(material_uri)
                    material_count += 1

                graph.add((dataset_uri, DATA.containsMaterial, material_uri))

                for column in columns:
                    value = row.get(column)
                    if value is None or value == "":
                        continue
                    prop_slug = slugify(column)
                    if model_slug:
                        obs_uri = TCKGMATI[f"{material_slug}_{model_slug}_{prop_slug}_property"]
                        prop_class = TCKGMAT.PredictedMaterialProperty
                    else:
                        obs_uri = TCKGMATI[f"{material_slug}_{prop_slug}_property"]
                        prop_class = TCKGMAT.ComputedMaterialProperty

                    add_named(graph, obs_uri, prop_class, f"{material_id} {column}")
                    graph.add((material_uri, DATA.hasMaterialProperty, obs_uri))
                    graph.add((obs_uri, DATA.propertyFromDataset, dataset_uri))
                    graph.add((obs_uri, DATA.propertyAttribute, add_property_attribute(graph, column)))
                    graph.add((obs_uri, DATA.hasPropertyName, Literal(column, datatype=XSD.string)))
                    graph.add((obs_uri, DATA.hasPropertyValue, literal_from_string(value)))
                    unit = unit_for_property(column)
                    if unit:
                        graph.add((obs_uri, DATA.hasPropertyUnit, Literal(unit, datatype=XSD.string)))
                    if model_variant_uri is not None:
                        graph.add((obs_uri, DATA.predictedByModelVariant, model_variant_uri))
                    graph.add((obs_uri, DCTERMS.source, Literal(str(path.relative_to(BASE_DIR)), datatype=XSD.string)))
                    property_count += 1

                rows_seen += 1
                if limit_per_file is not None and rows_seen >= limit_per_file:
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    return len(files), material_count, property_count


def parse_args():
    parser = argparse.ArgumentParser(description="Generate materials/properties TTL from dataset prediction files.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-file", type=int, default=1000)
    parser.add_argument("--files-limit", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Read all rows from all files.")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.full else args.limit_per_file
    files, materials, properties = generate(
        args.dataset_dir.resolve(),
        args.output.resolve(),
        limit_per_file=limit,
        files_limit=args.files_limit,
    )
    print(f"Generated {args.output}")
    print(f"files={files} materials={materials} material_properties={properties} limit_per_file={limit}")


if __name__ == "__main__":
    main()
