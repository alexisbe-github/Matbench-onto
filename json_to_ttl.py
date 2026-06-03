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

EVAL = Namespace("https://k.loria.fr/ontologies/evaluationonto#")
EVALIND = Namespace("https://k.loria.fr/ontologies/evaluationonto-individuals#")

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


def dataset_class_or_default(class_name):
    class_local_name = local_name(class_name)

    if class_local_name == "TrainingDataset":
        return TRAIN.TrainingDataset

    if not class_local_name:
        return DATA.Dataset

    return DATA[class_local_name]


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

    yielded_names = set()

    for entry in architectures.get("architectures", []):
        if isinstance(entry, dict):
            name = get_value(entry.get("name")) or get_value(entry.get("id"))
            data = entry
        else:
            name = entry
            data = architectures.get(name)

        if name and isinstance(data, dict):
            yielded_names.add(name)
            yield name, data

    for name, data in architectures.items():
        if name in {"architectures", "architecture_loop"}:
            continue
        if name in yielded_names:
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

    yielded_names = set()
    hyperparameters_field = architecture.get("hyperparameters", [])

    for name in normalize_list(hyperparameters_field):
        name = get_value(name)
        if not name:
            continue
        data = architecture.get(name)
        if not isinstance(data, dict):
            data = {"evidence": get_evidence(hyperparameters_field)}
        yielded_names.add(name)
        yield name, data

    for name, data in architecture.items():
        if name in {
            "ontology_class",
            "parameter_number",
            "datasets",
            "hyperparameters",
            "hyperparameter_loop",
            "components",
            "backbone_architecture",
            "head_architecture",
            "layer_count",
            "hidden_dimension",
        }:
            continue
        if name in yielded_names:
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

    yielded_names = set()

    for entry in training.get("training_runs", []):
        if isinstance(entry, dict):
            name = get_value(entry.get("id")) or get_value(entry.get("name"))
            data = entry
        else:
            name = entry
            data = training.get(name)

        if name and isinstance(data, dict):
            yielded_names.add(name)
            yield name, data

    for name, data in training.items():
        if name in {"training_runs", "training_run_loop"}:
            continue
        if name in yielded_names:
            continue
        if isinstance(data, dict):
            yield name, data


def iter_datasets(datasets):
    if not isinstance(datasets, dict):
        return

    if isinstance(datasets.get("dataset_loop"), dict):
        for name, data in datasets["dataset_loop"].items():
            if isinstance(data, dict):
                yield name, data
        return

    yielded_names = set()

    for entry in datasets.get("datasets", []):
        if isinstance(entry, dict):
            name = get_value(entry.get("name")) or get_value(entry.get("id"))
            data = entry
        else:
            name = entry
            data = datasets.get(name)

        if name and isinstance(data, dict):
            yielded_names.add(name)
            yield name, data

    for name, data in datasets.items():
        if name in {"datasets", "dataset_loop"}:
            continue
        if name in yielded_names:
            continue
        if isinstance(data, dict):
            yield name, data


def iter_benchmark_tasks(evaluation):
    if not isinstance(evaluation, dict):
        return

    if isinstance(evaluation.get("benchmark_task_loop"), dict):
        for name, data in evaluation["benchmark_task_loop"].items():
            if isinstance(data, dict):
                yield name, data
        return

    yielded_names = set()

    for entry in evaluation.get("benchmark_tasks", []):
        if isinstance(entry, dict):
            name = get_value(entry.get("name")) or get_value(entry.get("id"))
            data = entry
        else:
            name = entry
            data = evaluation.get(name)

        if name and isinstance(data, dict):
            yielded_names.add(name)
            yield name, data

    for name, data in evaluation.items():
        if name in {"benchmark_tasks", "benchmark_task_loop"}:
            continue
        if name in yielded_names:
            continue
        if isinstance(data, dict):
            yield name, data


def add_dataset_reference(graph, dataset_name, dataset_class=DATA.Dataset):
    dataset_name = get_value(dataset_name)

    if not dataset_name:
        return None

    dataset_uri = DATAIND[slugify(dataset_name)]
    add_named_individual(graph, dataset_uri, dataset_class, dataset_name)
    return dataset_uri


def add_model_materialization(graph, name, rdf_class=TRAIN.ModelMaterialization):
    name = get_value(name)

    if not name:
        return None

    materialization_uri = TRIND[f"{slugify(name)}_materialization"]
    add_named_individual(graph, materialization_uri, rdf_class, name)
    return materialization_uri


def add_algorithm_reference(graph, name, rdf_class):
    name = get_value(name)

    if not name:
        return None

    uri = TRIND[slugify(name)]
    add_named_individual(graph, uri, rdf_class, name)
    return uri


def build_graph(data):
    graph = Graph()
    rdf_star_lines = []

    graph.bind("arch", ARCH)
    graph.bind("archind", ARCHIND)
    graph.bind("train", TRAIN)
    graph.bind("trind", TRIND)
    graph.bind("data", DATA)
    graph.bind("dataind", DATAIND)
    graph.bind("eval", EVAL)
    graph.bind("evalind", EVALIND)
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
    datasets = data.get("datasets") or {"datasets": training.get("datasets", [])}
    evaluation = data.get("evaluation", {})

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
    architecture_hyperparameter_value_uris = {}

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
        architecture_hyperparameter_value_uris[architecture_name] = []

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

        layer_count_field = architecture.get("layer_count")
        layer_count = get_value(layer_count_field)

        if layer_count is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                architecture_config_uri,
                ARCH.hasLayerCount,
                literal_from_value(layer_count),
                layer_count_field,
                pdf_url
            )

        hidden_dimension_field = architecture.get("hidden_dimension")
        hidden_dimension = get_value(hidden_dimension_field)

        if hidden_dimension is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                architecture_config_uri,
                ARCH.hasHiddenDimension,
                literal_from_value(hidden_dimension),
                hidden_dimension_field,
                pdf_url
            )

        component_fields = [
            ("components", ARCH.hasComponent),
            ("backbone_architecture", ARCH.hasBackboneArchitecture),
            ("head_architecture", ARCH.hasHeadArchitecture),
        ]

        for field_name, predicate in component_fields:
            component_field = architecture.get(field_name)

            for component_name in normalize_list(component_field):
                component_name = get_value(component_name)

                if not component_name:
                    continue

                component_uri = ARCHIND[
                    f"{architecture_slug}_{slugify(component_name)}_component"
                ]
                component_config_uri = ARCHIND[
                    f"{architecture_slug}_{slugify(component_name)}_component_configuration"
                ]
                add_named_individual(
                    graph,
                    component_uri,
                    ARCH.MachineLearningArchitecture,
                    component_name
                )
                add_named_individual(
                    graph,
                    component_config_uri,
                    ARCH.MachineLearningArchitectureConfiguration,
                    f"{component_name} component configuration"
                )
                graph.add((
                    component_uri,
                    ARCH.hasMachineLearningArchitectureConfiguration,
                    component_config_uri
                ))

                add_triple_with_evidence(
                    graph, rdf_star_lines,
                    architecture_uri,
                    predicate,
                    component_uri,
                    component_field,
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

            architecture_hyperparameter_value_uris[architecture_name].append(
                (hp_value_uri, hp_data)
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

            for hp_value_uri, hp_data in architecture_hyperparameter_value_uris.get(
                architecture_name,
                []
            ):
                add_triple_with_evidence(
                    graph, rdf_star_lines,
                    run_uri,
                    TRAIN.hasHyperparameterValue,
                    hp_value_uri,
                    hp_data,
                    pdf_url
                )

        datasets_field = run_data.get("datasets")

        for dataset_name in normalize_list(datasets_field):
            dataset_uri = add_dataset_reference(graph, dataset_name)

            if dataset_uri is None:
                continue

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
            optimizer_uri = add_algorithm_reference(graph, optimizer, TRAIN.Optimizer)

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
            loss_uri = add_algorithm_reference(
                graph,
                loss_function,
                TRAIN.LossFunction
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesLoss,
                loss_uri,
                loss_field,
                pdf_url
            )

        objective_function_field = run_data.get("objective_function")
        objective_function = get_value(objective_function_field)

        if objective_function:
            objective_uri = add_algorithm_reference(
                graph,
                objective_function,
                TRAIN.ObjectiveFunction
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesObjectiveFunction,
                objective_uri,
                objective_function_field,
                pdf_url
            )

        sampling_method_field = run_data.get("sampling_method")
        sampling_method = get_value(sampling_method_field)

        if sampling_method:
            sampling_uri = add_algorithm_reference(
                graph,
                sampling_method,
                TRAIN.SamplingMethod
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesSamplingMethod,
                sampling_uri,
                sampling_method_field,
                pdf_url
            )

        uses_batched_selection_field = run_data.get("uses_batched_selection")
        uses_batched_selection = get_value(uses_batched_selection_field)

        if uses_batched_selection is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.usesBatchedSelection,
                literal_from_value(uses_batched_selection),
                uses_batched_selection_field,
                pdf_url
            )

        initialized_from_field = run_data.get("initialized_from_model_materialization")
        initialized_from_uri = add_model_materialization(graph, initialized_from_field)

        if initialized_from_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.initializedFromModelMaterialization,
                initialized_from_uri,
                initialized_from_field,
                pdf_url
            )

        teacher_model_field = run_data.get("teacher_model")
        teacher_model_uri = add_model_materialization(graph, teacher_model_field)

        if teacher_model_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.hasTeacherModel,
                teacher_model_uri,
                teacher_model_field,
                pdf_url
            )

        student_model_field = run_data.get("student_model")
        student_model_uri = add_model_materialization(graph, student_model_field)

        if student_model_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                TRAIN.hasStudentModel,
                student_model_uri,
                student_model_field,
                pdf_url
            )

        materializations_field = run_data.get("model_materializations")

        for materialization_name in normalize_list(materializations_field):
            materialization_uri = add_model_materialization(
                graph,
                materialization_name,
                TRAIN.Checkpoint
            )

            if materialization_uri is None:
                continue

            add_triple_with_evidence(
                graph, rdf_star_lines,
                materialization_uri,
                TRAIN.generatedBy,
                run_uri,
                materializations_field,
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

    for dataset_name, dataset_data in iter_datasets(datasets):
        dataset_class = dataset_class_or_default(dataset_data.get("ontology_class"))
        dataset_uri = add_dataset_reference(graph, dataset_name, dataset_class)

        if dataset_uri is None:
            continue

        samples_field = dataset_data.get("number_of_samples")
        samples = get_value(samples_field)

        if samples is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.hasNumberOfSamples,
                literal_from_value(samples),
                samples_field,
                pdf_url
            )

        representation_field = dataset_data.get("data_representation")
        representation = get_value(representation_field)

        if representation:
            representation_uri = DATAIND[f"{slugify(representation)}_representation"]
            add_named_individual(
                graph,
                representation_uri,
                DATA.DataRepresentation,
                representation
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.hasDataRepresentation,
                representation_uri,
                representation_field,
                pdf_url
            )

        labelling_method_field = dataset_data.get("labelling_method")
        labelling_method = get_value(labelling_method_field)

        if labelling_method:
            labelling_method_uri = DATAIND[
                f"{slugify(labelling_method)}_labelling_method"
            ]
            add_named_individual(
                graph,
                labelling_method_uri,
                DATA.LabellingMethod,
                labelling_method
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.hasLabellingMethod,
                labelling_method_uri,
                labelling_method_field,
                pdf_url
            )

        input_features_field = dataset_data.get("input_features")

        for feature_name in normalize_list(input_features_field):
            feature_name = get_value(feature_name)

            if not feature_name:
                continue

            feature_uri = DATAIND[f"{slugify(feature_name)}_attribute"]
            add_named_individual(graph, feature_uri, DATA.DatasetAttribute, feature_name)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.usesAttributeAsInputFeature,
                feature_uri,
                input_features_field,
                pdf_url
            )

        target_features_field = dataset_data.get("target_features")

        for feature_name in normalize_list(target_features_field):
            feature_name = get_value(feature_name)

            if not feature_name:
                continue

            feature_uri = DATAIND[f"{slugify(feature_name)}_attribute"]
            add_named_individual(graph, feature_uri, DATA.DatasetAttribute, feature_name)

            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.usesAttributeAsTargetFeature,
                feature_uri,
                target_features_field,
                pdf_url
            )

        derived_from_field = dataset_data.get("derived_from_dataset")
        derived_from_uri = add_dataset_reference(graph, derived_from_field)

        if derived_from_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                dataset_uri,
                DATA.wasDerivedFromDataset,
                derived_from_uri,
                derived_from_field,
                pdf_url
            )

    benchmark_suite_field = evaluation.get("benchmark_suite")
    benchmark_suite = get_value(benchmark_suite_field)
    benchmark_release_field = evaluation.get("benchmark_release")
    benchmark_release = get_value(benchmark_release_field)
    benchmark_release_uri = None

    if benchmark_suite:
        benchmark_suite_uri = EVALIND[f"{slugify(benchmark_suite)}_benchmark_suite"]
        add_named_individual(
            graph,
            benchmark_suite_uri,
            EVAL.BenchmarkSuite,
            benchmark_suite
        )

    if benchmark_release:
        benchmark_release_uri = EVALIND[
            f"{slugify(benchmark_release)}_benchmark_release"
        ]
        add_named_individual(
            graph,
            benchmark_release_uri,
            EVAL.BenchmarkRelease,
            benchmark_release
        )

    for task_name, task_data in iter_benchmark_tasks(evaluation):
        task_uri = EVALIND[f"{slugify(task_name)}_benchmark_task"]
        result_uri = EVALIND[
            f"{slugify(variant_name)}_{slugify(task_name)}_benchmark_result"
        ]

        add_named_individual(graph, task_uri, EVAL.BenchmarkTask, task_name)
        add_named_individual(
            graph,
            result_uri,
            EVAL.BenchmarkResult,
            f"{variant_name} {task_name} benchmark result"
        )

        if benchmark_release_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                benchmark_release_uri,
                EVAL.hasTask,
                task_uri,
                evaluation.get("benchmark_tasks"),
                pdf_url
            )

        task_dataset_field = task_data.get("dataset")
        task_dataset_uri = add_dataset_reference(graph, task_dataset_field)

        if task_dataset_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                result_uri,
                EVAL.usesDataset,
                task_dataset_uri,
                task_dataset_field,
                pdf_url
            )

        fold_uris = []

        for index, fold_name in enumerate(normalize_list(task_data.get("folds"))):
            fold_name = get_value(fold_name)

            if not fold_name:
                continue

            fold_uri = EVALIND[f"{slugify(task_name)}_{slugify(fold_name)}_fold"]
            add_named_individual(graph, fold_uri, EVAL.Fold, fold_name)
            graph.add((fold_uri, EVAL.hasIndex, Literal(index, datatype=XSD.integer)))
            fold_uris.append(fold_uri)

        for metric in normalize_list(task_data.get("metric_results")):
            metric_value = get_value(metric)

            if not metric_value:
                continue

            if isinstance(metric_value, dict):
                metric_name = get_value(metric_value.get("metric_name")) or "metric"
                value = get_value(metric_value.get("value"))
                unit = get_value(metric_value.get("unit"))
                metric_label = " ".join(
                    str(part) for part in (metric_name, value, unit) if part is not None
                )
            else:
                metric_name = str(metric_value)
                metric_label = str(metric_value)

            metric_uri = EVALIND[
                f"{slugify(variant_name)}_{slugify(task_name)}_{slugify(metric_name)}_metric_result"
            ]
            add_named_individual(graph, metric_uri, EVAL.MetricResult, metric_label)

            for fold_uri in fold_uris:
                add_triple_with_evidence(
                    graph, rdf_star_lines,
                    fold_uri,
                    EVAL.producesMetricResult,
                    metric_uri,
                    metric,
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
