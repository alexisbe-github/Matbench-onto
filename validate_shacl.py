import argparse
import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace, RDF, OWL, RDFS, URIRef


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TTL_DIR = BASE_DIR / "outputs" / "ttl"
ONTOLOGY_FILES = [
    BASE_DIR / "ontology" / "architecture.ttl",
    BASE_DIR / "ontology" / "trainingonto.ttl",
    BASE_DIR / "ontology" / "datasetonto.ttl",
]

ARCH = Namespace("https://k.loria.fr/ontologies/architectureonto#")
TRAIN = Namespace("https://k.loria.fr/ontologies/trainingonto#")
DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")

ONTOLOGY_NAMESPACES = (
    str(ARCH),
    str(TRAIN),
    str(DATA),
)

SHACL_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

@prefix arch: <https://k.loria.fr/ontologies/architectureonto#> .
@prefix train: <https://k.loria.fr/ontologies/trainingonto#> .
@prefix data: <https://k.loria.fr/ontologies/datasetonto#> .

# Ensure that only training runs can use datasets.

train:UsesDatasetSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesDataset ;
    sh:class train:MachineLearningTrainingRun ;
    sh:message "train:usesDataset must have a MachineLearningTrainingRun as subject." .

train:UsesDatasetObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesDataset ;
    sh:class data:Dataset ;
    sh:message "train:usesDataset must point to a data:Dataset." .


# Ensure that training runs train model variants.

train:TrainsModelVariantObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:trainsModelVariant ;
    sh:class arch:ModelVariant ;
    sh:message "train:trainsModelVariant must point to an arch:ModelVariant." .


# Ensure that every model variant is linked to an architecture.

arch:ModelVariantMustHaveArchitectureShape
    a sh:NodeShape ;
    sh:targetClass arch:ModelVariant ;
    sh:property [
        sh:path arch:hasMachineLearningArchitecture ;
        sh:minCount 1 ;
        sh:class arch:MachineLearningArchitecture ;
        sh:message "every ModelVariant should link to a MachineLearningArchitecture." ;
    ] .


# Ensure that a ModelFamily only contains ModelVariant instances.

arch:HasVariantSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasVariant ;
    sh:class arch:ModelFamily ;
    sh:message "arch:hasVariant must have a ModelFamily as subject." .

arch:HasVariantObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasVariant ;
    sh:class arch:ModelVariant ;
    sh:message "arch:hasVariant must point to a ModelVariant." .


# Ensure that ModelVariant links point to architectures.

arch:HasArchitectureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasMachineLearningArchitecture ;
    sh:class arch:ModelVariant ;
    sh:message "arch:hasMachineLearningArchitecture must have a ModelVariant as subject." .

arch:HasArchitectureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasMachineLearningArchitecture ;
    sh:class arch:MachineLearningArchitecture ;
    sh:message "arch:hasMachineLearningArchitecture must point to a MachineLearningArchitecture." .


# Ensure that architectures point to architecture configurations.

arch:HasArchitectureConfigurationSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasMachineLearningArchitectureConfiguration ;
    sh:class arch:MachineLearningArchitecture ;
    sh:message "only a MachineLearningArchitecture should have an architecture configuration." .

arch:HasArchitectureConfigurationObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasMachineLearningArchitectureConfiguration ;
    sh:class arch:MachineLearningArchitectureConfiguration ;
    sh:message "arch:hasMachineLearningArchitectureConfiguration must point to a MachineLearningArchitectureConfiguration." .


# Ensure that optimizers are Optimizer instances.

train:UsesOptimizerShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesOptimizer ;
    sh:class train:Optimizer ;
    sh:message "train:usesOptimizer must point to an Optimizer." .


# Ensure that losses are LossFunction instances.

train:UsesLossShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesLoss ;
    sh:class train:LossFunction ;
    sh:message "train:usesLoss must point to a LossFunction." .


# Ensure that objective functions are ObjectiveFunction instances.

train:UsesObjectiveFunctionShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesObjectiveFunction ;
    sh:class train:ObjectiveFunction ;
    sh:message "train:usesObjectiveFunction must point to an ObjectiveFunction." .


# Ensure that sampling methods are SamplingMethod instances.

train:UsesSamplingMethodShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesSamplingMethod ;
    sh:class train:SamplingMethod ;
    sh:message "train:usesSamplingMethod must point to a SamplingMethod." .


# Ensure that parameter numbers are non-negative integers.

arch:ParameterNumberShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasParameterNumber ;
    sh:property [
        sh:path arch:hasParameterNumber ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "arch:hasParameterNumber must be a non-negative integer." ;
    ] .


# Ensure that layer counts are non-negative integers.

arch:LayerCountShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasLayerCount ;
    sh:property [
        sh:path arch:hasLayerCount ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "arch:hasLayerCount must be a non-negative integer." ;
    ] .


# Ensure that hidden dimensions are positive integers.

arch:HiddenDimensionShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasHiddenDimension ;
    sh:property [
        sh:path arch:hasHiddenDimension ;
        sh:datatype xsd:integer ;
        sh:minInclusive 1 ;
        sh:message "arch:hasHiddenDimension must be a positive integer." ;
    ] .

    
# Ensure that parameter number is positive integer
    
arch:ParameterNumberShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasParameterNumber ;
    sh:property [
        sh:path arch:hasParameterNumber ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "LLM error: arch:hasParameterNumber must be a non-negative integer." ;
    ] .
"""


def strip_rdf_star_lines(ttl_text):
    return "\n".join(
        line for line in ttl_text.splitlines()
        if not line.lstrip().startswith("<<")
    )


def parse_turtle_file(path, strip_rdf_star=False):
    graph = Graph()
    text = path.read_text(encoding="utf-8")

    if strip_rdf_star:
        text = strip_rdf_star_lines(text)

    graph.parse(data=text, format="turtle", publicID=path.as_uri())
    return graph


def load_ontology_graph():
    graph = Graph()

    for path in ONTOLOGY_FILES:
        if path.exists():
            graph.parse(path, format="turtle")

    return graph


def load_shapes_graph():
    graph = Graph()
    graph.parse(data=SHACL_SHAPES, format="turtle")
    return graph


def known_ontology_classes(ontology_graph):
    classes = set()

    for class_type in (OWL.Class, RDFS.Class):
        classes.update(ontology_graph.subjects(RDF.type, class_type))

    return classes


def find_unknown_ontology_types(data_graph, ontology_graph):
    known_classes = known_ontology_classes(ontology_graph)
    unknown_types = set()

    for _, _, rdf_type in data_graph.triples((None, RDF.type, None)):
        if not isinstance(rdf_type, URIRef):
            continue

        rdf_type_text = str(rdf_type)

        if rdf_type == OWL.NamedIndividual:
            continue

        if rdf_type_text.startswith(ONTOLOGY_NAMESPACES) and rdf_type not in known_classes:
            unknown_types.add(rdf_type)

    return sorted(unknown_types, key=str)

def get_shacl_report(ttl_path):
    ttl_path = Path(ttl_path)

    data_graph = parse_turtle_file(ttl_path, strip_rdf_star=True)
    ontology_graph = load_ontology_graph()
    shapes_graph = load_shapes_graph()

    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=True,
        serialize_report_graph=False,
    )

    unknown_types = find_unknown_ontology_types(
        data_graph,
        ontology_graph,
    )

    return {
        "ttl_path": str(ttl_path),
        "conforms": conforms and not unknown_types,
        "shacl_conforms": conforms,
        "report_graph": report_graph,
        "report_text": report_text,
        "unknown_types": [str(x) for x in unknown_types],
    }

def validate_ttl_file(ttl_path, report_dir=None):
    print(f"\n=== VALIDATE SHACL ===")
    print(f"TTL: {ttl_path}")

    data_graph = parse_turtle_file(ttl_path, strip_rdf_star=True)
    ontology_graph = load_ontology_graph()
    shapes_graph = load_shapes_graph()

    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=True,
        serialize_report_graph=False,
    )

    unknown_types = find_unknown_ontology_types(data_graph, ontology_graph)

    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        report_txt_path = report_dir / f"{ttl_path.stem}_shacl_report.txt"
        report_txt_path.write_text(report_text, encoding="utf-8")

        report_ttl_path = report_dir / f"{ttl_path.stem}_shacl_report.ttl"
        report_graph.serialize(destination=report_ttl_path, format="turtle")

    if conforms and not unknown_types:
        print("[OK] SHACL conforms")
        return True

    if not conforms:
        print("[FAIL] SHACL validation failed")
        print(report_text)

    if unknown_types:
        print("[FAIL] Unknown ontology classes used as rdf:type:")
        for rdf_type in unknown_types:
            print(f"  - {rdf_type}")

    return False


def ttl_files_from_args(args):
    if args.ttl:
        return [Path(args.ttl).resolve()]

    env_ttl = args.env_ttl
    if env_ttl:
        return [Path(env_ttl).resolve()]

    return sorted(DEFAULT_TTL_DIR.glob("*.ttl"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate generated Matbench ontology TTL files with SHACL."
    )
    parser.add_argument(
        "--ttl",
        help="Path to one generated TTL file. Defaults to INPUT_TTL_FILE or outputs/ttl/*.ttl.",
    )
    parser.add_argument(
        "--env-ttl",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
    "--report-dir",
    default=str(BASE_DIR / "outputs" / "shacl_reports"),
    help="Directory where SHACL reports are written.",
    )   
    return parser.parse_args()


def main():
    args = parse_args()

    if args.env_ttl is None:
        import os
        args.env_ttl = os.getenv("INPUT_TTL_FILE")

    ttl_files = ttl_files_from_args(args)

    if not ttl_files:
        print(f"No TTL files found in {DEFAULT_TTL_DIR}")
        return 1

    failed = []

    for ttl_path in ttl_files:
        try:
            if not validate_ttl_file(ttl_path, report_dir=args.report_dir):
                failed.append(ttl_path)
        except Exception as error:
            print(f"\n[ERROR] {ttl_path}")
            print(error)
            failed.append(ttl_path)

    if failed:
        print(f"\nValidation failed for {len(failed)} file(s).")
        return 1

    print(f"\nValidated {len(ttl_files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
