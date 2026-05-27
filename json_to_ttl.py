import json
import os
import re
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef
from rdflib.namespace import XSD, PROV


BASE_DIR = Path(__file__).resolve().parent

INPUT_JSON_FILE = Path(
    os.getenv("INPUT_JSON_FILE", BASE_DIR / "outputs" / "model_extraction.json")
)

OUTPUT_TTL_FILE = Path(
    os.getenv("OUTPUT_TTL_FILE", BASE_DIR / "outputs" / "model_individuals_generated.ttl")
)

ARCH = Namespace("https://k.loria.fr/ontologies/architectureonto#")
ARCHIND = Namespace("https://k.loria.fr/ontologies/architectureonto-individuals#")

TRAIN = Namespace("https://k.loria.fr/ontologies/trainingonto#")
TRIND = Namespace("https://k.loria.fr/ontologies/trainingonto-individuals#")

DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")
DATAIND = Namespace("https://k.loria.fr/ontologies/dataset-individuals#")

DCTERMS = Namespace("http://purl.org/dc/terms/")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_value(field):
    if isinstance(field, dict) and "value" in field:
        return field.get("value")
    return field


def get_evidence(field):
    if isinstance(field, dict):
        return field.get("evidence")
    return None


def slugify(value):
    value = get_value(value)

    if value is None:
        return "unknown"

    value = str(value).strip().lower()
    value = value.replace("Å", "angstrom").replace("å", "angstrom")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value or "unknown"


def local_name(value):
    value = get_value(value)

    if not value:
        return None

    return str(value).strip().split(":")[-1]


def literal_from_value(value):
    value = get_value(value)

    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)

    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)

    if isinstance(value, float):
        return Literal(value, datatype=XSD.float)

    text = str(value)

    if re.fullmatch(r"-?\d+", text):
        return Literal(int(text), datatype=XSD.integer)

    if re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", text):
        return Literal(float(text), datatype=XSD.float)

    return Literal(text, datatype=XSD.string)


def normalize_list(value):
    value = get_value(value)

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def add_named_individual(graph, uri, rdf_class, label=None):
    graph.add((uri, RDF.type, OWL.NamedIndividual))
    graph.add((uri, RDF.type, rdf_class))

    label = get_value(label)

    if label:
        graph.add((uri, RDFS.label, Literal(str(label), datatype=XSD.string)))


def class_or_default(namespace, class_name, default_class):
    class_local_name = local_name(class_name)

    if not class_local_name:
        return default_class

    return namespace[class_local_name]


def add_triple_with_evidence(
    graph,
    rdf_star_lines,
    subject,
    predicate,
    obj,
    evidence_source=None,
    pdf_url=None,
):
    graph.add((subject, predicate, obj))

    evidence = get_evidence(evidence_source) or evidence_source

    if not evidence:
        return

    escaped = str(evidence).replace("\\", "\\\\").replace('"', '\\"')

    line = (
        f"<< {subject.n3()} {predicate.n3()} {obj.n3()} >> "
        f'prov:wasDerivedFrom "{escaped}"'
    )

    if pdf_url:
        safe_pdf_url = str(pdf_url).replace("\\", "\\\\").replace('"', '\\"')
        line += f" ; dcterms:source <{safe_pdf_url}>"

    line += " ."

    rdf_star_lines.append(line)


def iter_architectures(architectures):
    if not isinstance(architectures, dict):
        return

    if isinstance(architectures.get("architecture_loop"), dict):
        for name, data in architectures["architecture_loop"].items():
            if isinstance(data, dict):
                yield name, data
        return

    for name in architectures.get("architectures", []):
        data = architectures.get(name)
        if isinstance(data, dict):
            yield name, data

    for name, data in architectures.items():
        if name in {"architectures", "architecture_loop"}:
            continue
        if isinstance(data, dict):
            yield name, data


def iter_hyperparameters(architecture):
    if not isinstance(architecture, dict):
        return

    if isinstance(architecture.get("hyperparameter_loop"), dict):
        for name, data in architecture["hyperparameter_loop"].items():
            if isinstance(data, dict):
                yield name, data
        return

    for name in architecture.get("hyperparameters", []):
        data = architecture.get(name)
        if isinstance(data, dict):
            yield name, data

    for name, data in architecture.items():
        if name in {
            "ontology_class",
            "parameter_number",
            "datasets",
            "hyperparameters",
            "hyperparameter_loop",
        }:
            continue
        if isinstance(data, dict) and "value" in data:
            yield name, data


def iter_training_runs(training):
    if not isinstance(training, dict):
        return

    if isinstance(training.get("training_run_loop"), dict):
        for name, data in training["training_run_loop"].items():
            if isinstance(data, dict):
                yield name, data
        return

    for name in training.get("training_runs", []):
        data = training.get(name)
        if isinstance(data, dict):
            yield name, data

    for name, data in training.items():
        if name in {"training_runs", "training_run_loop"}:
            continue
        if isinstance(data, dict):
            yield name, data


def build_graph(data):
    graph = Graph()
    rdf_star_lines = []

    graph.bind("arch", ARCH)
    graph.bind("archind", ARCHIND)
    graph.bind("train", TRAIN)
    graph.bind("trind", TRIND)
    graph.bind("data", DATA)
    graph.bind("dataind", DATAIND)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("owl", OWL)
    graph.bind("xsd", XSD)
    graph.bind("prov", PROV)
    graph.bind("dcterms", DCTERMS)

    sources = data.get("_sources", {})
    pdf_url = sources.get("pdf_url")
    yaml_file = sources.get("yaml_file")

    model = data.get("model", {})
    architectures = data.get("architectures", {})
    training = data.get("training", {})

    family_field = model.get("family")
    variant_field = model.get("variant")
    parameter_field = model.get("parameter_number")

    family_name = get_value(family_field)
    variant_name = get_value(variant_field)
    model_parameter_number = get_value(parameter_field)

    family_uri = ARCHIND[f"{slugify(family_name)}_family"]
    variant_uri = ARCHIND[f"{slugify(variant_name)}_variant"]

    add_named_individual(graph, family_uri, ARCH.ModelFamily, family_name)
    add_named_individual(graph, variant_uri, ARCH.ModelVariant, variant_name)

    if pdf_url:
        graph.add((variant_uri, DCTERMS.source, URIRef(pdf_url)))

    if yaml_file:
        graph.add((variant_uri, PROV.wasDerivedFrom, Literal(str(yaml_file))))

    add_triple_with_evidence(
        graph, rdf_star_lines,
        family_uri, ARCH.hasVariant, variant_uri,
        variant_field, pdf_url
    )

    if model_parameter_number is not None:
        add_triple_with_evidence(
            graph, rdf_star_lines,
            variant_uri, ARCH.hasParameterNumber,
            literal_from_value(model_parameter_number),
            parameter_field, pdf_url
        )

    architecture_variant_uris = {}

    for architecture_name, architecture in iter_architectures(architectures):
        architecture_slug = slugify(architecture_name)
        variant_slug = slugify(variant_name)

        architecture_class_field = architecture.get("ontology_class")
        architecture_class = class_or_default(
            ARCH,
            architecture_class_field,
            ARCH.MachineLearningArchitecture
        )

        architecture_uri = ARCHIND[f"{variant_slug}_{architecture_slug}_architecture"]
        architecture_config_uri = ARCHIND[f"{variant_slug}_{architecture_slug}_architecture_configuration"]
        architecture_variant_uri = ARCHIND[f"{variant_slug}_{architecture_slug}_model_variant"]

        architecture_variant_uris[architecture_name] = architecture_variant_uri

        add_named_individual(graph, architecture_uri, architecture_class, architecture_name)
        add_named_individual(
            graph,
            architecture_config_uri,
            ARCH.MachineLearningArchitectureConfiguration,
            f"{architecture_name} architecture configuration"
        )
        add_named_individual(
            graph,
            architecture_variant_uri,
            ARCH.ModelVariant,
            f"{variant_name} {architecture_name}"
        )

        add_triple_with_evidence(
            graph, rdf_star_lines,
            family_uri, ARCH.hasVariant, architecture_variant_uri,
            architecture_class_field, pdf_url
        )

        add_triple_with_evidence(
            graph, rdf_star_lines,
            architecture_variant_uri,
            ARCH.hasMachineLearningArchitecture,
            architecture_uri,
            architecture_class_field,
            pdf_url
        )

        graph.add((
            architecture_uri,
            ARCH.hasMachineLearningArchitectureConfiguration,
            architecture_config_uri
        ))

        architecture_parameter_field = architecture.get("parameter_number")
        architecture_parameter_number = get_value(architecture_parameter_field)

        if architecture_parameter_number is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                architecture_uri,
                ARCH.hasParameterNumber,
                literal_from_value(architecture_parameter_number),
                architecture_parameter_field,
                pdf_url
            )

        datasets_field = architecture.get("datasets")

        for dataset_name in normalize_list(datasets_field):
            dataset_name = get_value(dataset_name)

            if not dataset_name:
                continue

            dataset_uri = DATAIND[slugify(dataset_name)]
            add_named_individual(graph, dataset_uri, DATA.Dataset, dataset_name)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                architecture_uri,
                TRAIN.usesDataset,
                dataset_uri,
                datasets_field,
                pdf_url
            )

        for hp_name, hp_data in iter_hyperparameters(architecture):
            hp_slug = slugify(hp_name)

            hp_uri = TRIND[hp_slug]
            hp_value_uri = TRIND[f"{slugify(variant_name)}_{architecture_slug}_{hp_slug}_value"]

            hp_class = class_or_default(
                TRAIN,
                hp_data.get("ontology_class"),
                TRAIN.MachineLearningHyperparameter
            )

            add_named_individual(graph, hp_uri, hp_class, hp_name)
            add_named_individual(
                graph,
                hp_value_uri,
                TRAIN.MachineLearningHyperparameterValue,
                f"{variant_name} {architecture_name} {hp_name} value"
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                hp_value_uri,
                TRAIN.ofHyperparameter,
                hp_uri,
                hp_data.get("ontology_class") or hp_data,
                pdf_url
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                architecture_config_uri,
                TRAIN.hasHyperparameterValue,
                hp_value_uri,
                hp_data,
                pdf_url
            )

            hp_value_field = hp_data.get("value")
            hp_value = get_value(hp_value_field)

            if hp_value is not None:
                add_triple_with_evidence(
                    graph, rdf_star_lines,
                    hp_value_uri,
                    PROV.value,
                    literal_from_value(hp_value),
                    hp_value_field,
                    pdf_url
                )

    for run_name, run_data in iter_training_runs(training):
        run_slug = slugify(run_name)

        run_type_field = run_data.get("type")
        run_class = class_or_default(
            TRAIN,
            run_type_field,
            TRAIN.MachineLearningTrainingRun
        )

        run_uri = TRIND[run_slug]

        add_named_individual(graph, run_uri, run_class, run_name)

        for architecture_name in normalize_list(run_data.get("architectures")):
            architecture_name = get_value(architecture_name)

            if not architecture_name:
                continue

            architecture_variant_uri = architecture_variant_uris.get(
                architecture_name,
                ARCHIND[f"{slugify(variant_name)}_{slugify(architecture_name)}_model_variant"]
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.trainsModelVariant,
                architecture_variant_uri,
                run_data.get("architectures"),
                pdf_url
            )

        datasets_field = run_data.get("datasets")

        for dataset_name in normalize_list(datasets_field):
            dataset_name = get_value(dataset_name)

            if not dataset_name:
                continue

            dataset_uri = DATAIND[slugify(dataset_name)]
            add_named_individual(graph, dataset_uri, DATA.Dataset, dataset_name)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesDataset,
                dataset_uri,
                datasets_field,
                pdf_url
            )

        optimizer_field = run_data.get("optimizer")
        optimizer = get_value(optimizer_field)

        if optimizer:
            optimizer_uri = TRIND[slugify(optimizer)]
            add_named_individual(graph, optimizer_uri, TRAIN.Optimizer, optimizer)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesOptimizer,
                optimizer_uri,
                optimizer_field,
                pdf_url
            )

        loss_field = run_data.get("loss_function")
        loss_function = get_value(loss_field)

        if loss_function:
            loss_uri = TRIND[slugify(loss_function)]
            add_named_individual(graph, loss_uri, TRAIN.LossFunction, loss_function)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesLoss,
                loss_uri,
                loss_field,
                pdf_url
            )

        description_field = run_data.get("description")
        description = get_value(description_field)

        if description:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                RDFS.comment,
                Literal(str(description), datatype=XSD.string),
                description_field,
                pdf_url
            )

    return graph, rdf_star_lines


data = load_json(INPUT_JSON_FILE)
graph, rdf_star_lines = build_graph(data)

OUTPUT_TTL_FILE.parent.mkdir(parents=True, exist_ok=True)

ttl_text = graph.serialize(format="turtle")

if rdf_star_lines:
    ttl_text += "\n\n"
    ttl_text += "\n".join(rdf_star_lines)
    ttl_text += "\n"

with open(OUTPUT_TTL_FILE, "w", encoding="utf-8") as file:
    file.write(ttl_text)

print(f"TTL généré : {OUTPUT_TTL_FILE}")