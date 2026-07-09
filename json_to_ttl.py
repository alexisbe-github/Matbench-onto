import json
import os
import re
from pathlib import Path

import yaml
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


TASK_ALIASES = {
    "thermodynamic_stability_classification_task": "thermodynamic_stability_classification_task",
    "thermodynamic_stability_classification": "thermodynamic_stability_classification_task",
    "stable_unstable_material_classification": "thermodynamic_stability_classification_task",
    "stable_unstable_classification": "thermodynamic_stability_classification_task",
    "stability_classification": "thermodynamic_stability_classification_task",
    "classifying_thermodynamic_stability": "thermodynamic_stability_classification_task",
    "classify_thermodynamic_stability": "thermodynamic_stability_classification_task",
    "discovery": "thermodynamic_stability_classification_task",
    "discovery_screening": "thermodynamic_stability_classification_task",
    "convex_hull_distance_regression_task": "convex_hull_distance_regression_task",
    "convex_hull_distance_regression": "convex_hull_distance_regression_task",
    "convex_hull_distance": "convex_hull_distance_regression_task",
    "hull_distance_regression": "convex_hull_distance_regression_task",
    "mae_of_predicted_vs_dft_convex_hull_distance": "convex_hull_distance_regression_task",
    "phonon_thermal_conductivity_contribution_task": "phonon_thermal_conductivity_contribution_task",
    "phonon_thermal_conductivity_contribution": "phonon_thermal_conductivity_contribution_task",
    "phonon_mode_contributions_to_thermal_conductivity": "phonon_thermal_conductivity_contribution_task",
    "phonons": "phonon_thermal_conductivity_contribution_task",
    "relaxed_structure_matching_rmsd_task": "relaxed_structure_matching_rmsd_task",
    "relaxed_structure_matching_rmsd": "relaxed_structure_matching_rmsd_task",
    "structurematcher_rmsd": "relaxed_structure_matching_rmsd_task",
    "structure_matching_rmsd": "relaxed_structure_matching_rmsd_task",
    "geo_opt": "relaxed_structure_matching_rmsd_task",
    "geometry_optimization": "relaxed_structure_matching_rmsd_task",
}


TASK_LABELS = {
    "thermodynamic_stability_classification_task": "Thermodynamic stability classification",
    "convex_hull_distance_regression_task": "Convex hull distance regression",
    "phonon_thermal_conductivity_contribution_task": "Phonon thermal conductivity contribution prediction",
    "relaxed_structure_matching_rmsd_task": "Relaxed structure matching RMSD",
}


TASK_CLASSES = {
    "thermodynamic_stability_classification_task": EVAL.ClassificationTask,
    "convex_hull_distance_regression_task": EVAL.RegressionTask,
    "phonon_thermal_conductivity_contribution_task": EVAL.RegressionTask,
    "relaxed_structure_matching_rmsd_task": EVAL.RegressionTask,
}

CLASSIFICATION_METRICS = {
    "F1",
    "DAF",
    "Precision",
    "Recall",
    "Accuracy",
    "TPR",
    "FPR",
    "TNR",
    "FNR",
    "TP",
    "FP",
    "TN",
    "FN",
}

CONVEX_HULL_REGRESSION_METRICS = {
    "MAE",
    "RMSE",
    "R2",
    "missing_preds",
}

NON_RESULT_KEYS = {
    "pred_file",
    "pred_file_url",
    "pred_col",
    "struct_col",
    "analysis_file",
    "analysis_file_url",
}

MATBENCH_TASKS = {
    "thermodynamic_stability_classification_task": {
        "type": "ClassificationTask",
        "dataset": "WBM test set",
    },
    "convex_hull_distance_regression_task": {
        "type": "RegressionTask",
        "dataset": "WBM test set",
    },
    "phonon_thermal_conductivity_contribution_task": {
        "type": "RegressionTask",
        "dataset": "kappa-103 phonon benchmark set",
    },
    "relaxed_structure_matching_rmsd_task": {
        "type": "RegressionTask",
        "dataset": "WBM test set",
    },
}


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


def get_source(field):
    if not isinstance(field, dict):
        return None

    source = field.get("source")

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return source

    return None


def slugify(value):
    value = get_value(value)

    if value is None:
        return "unknown"

    value = str(value).strip().lower()
    value = value.replace("κ", "kappa").replace("îº", "kappa")
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


def evidence_field(value, evidence):
    return {
        "value": value,
        "evidence": evidence,
    }


def has_meaningful_value(field):
    value = get_value(field)

    if value is None:
        return False

    if isinstance(value, str) and not value.strip():
        return False

    return True


def is_training_run_data(name, data):
    if not isinstance(data, dict):
        return False

    if name in {
        "datasets",
        "dataset_details",
        "evidence",
        "runs",
        "training_set",
        "training_sets",
        "crystal_structure_attribute",
    }:
        return False

    run_type = slugify(data.get("type"))
    if "run" in run_type or "training" in run_type or "finetuning" in run_type:
        return True

    return any(
        key in data
        for key in (
            "architectures",
            "model_variant",
            "datasets",
            "optimizer",
            "loss_function",
            "epochs",
            "batch_size",
            "learning_rate",
            "initialized_from",
            "initialized_from_model_materialization",
        )
    )


def metric_unit(metric_name):
    normalized = slugify(metric_name)

    if normalized in {"mae", "rmse"}:
        return "eV/atom"

    if normalized in {"tp", "fp", "tn", "fn", "missing_preds", "n_structures"}:
        return "count"

    if normalized in {
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tpr",
        "fpr",
        "tnr",
        "fnr",
        "symmetry_decrease",
        "symmetry_match",
        "symmetry_increase",
    }:
        return "fraction"

    return "dimensionless"


def iter_metric_leaf_values(node, path=()):
    if not isinstance(node, dict):
        return

    for key, value in node.items():
        if key in NON_RESULT_KEYS:
            continue

        if isinstance(value, dict):
            yield from iter_metric_leaf_values(value, (*path, key))
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield path, key, value


def init_evaluation_run(evaluation, run_id, task_id, model_variant, evidence):
    evaluation["evaluation_runs"][run_id] = {
        "type": evidence_field(
            "MachineLearningEvaluationRun",
            "ontology/evaluationonto.ttl: MachineLearningEvaluationRun",
        ),
        "model_variant": evidence_field(model_variant, evidence),
        "benchmark_release": evidence_field(
            "matbench_discovery_benchmark_release",
            evidence,
        ),
        "task": evidence_field(task_id, evidence),
        "dataset": evidence_field(MATBENCH_TASKS[task_id]["dataset"], evidence),
        "folds": evidence_field([], evidence),
        "metric_results": evidence_field([], evidence),
    }


def scoped_evaluation_id(model_variant, *parts):
    return "_".join(slugify(part) for part in (model_variant, *parts) if part)


def add_yaml_metric_result(
    evaluation,
    run_id,
    metric_family,
    split_path,
    metric_name,
    metric_value,
):
    split_id = "_".join(slugify(part) for part in split_path if part)
    metric_id_parts = [run_id, metric_family, split_id, metric_name]
    metric_id = "_".join(slugify(part) for part in metric_id_parts if part)
    evidence_path = ".".join(["YAML metrics", metric_family, *split_path, metric_name])
    run_data = evaluation["evaluation_runs"][run_id]

    evaluation["metric_results"][metric_id] = {
        "type": evidence_field("MetricResult", "ontology/evaluationonto.ttl: MetricResult"),
        "metric_type": evidence_field(metric_name, evidence_path),
        "metric_value": evidence_field(metric_value, evidence_path),
        "unit": evidence_field(metric_unit(metric_name), evidence_path),
        "task": run_data["task"],
        "dataset": evidence_field(split_id or metric_family, evidence_path),
    }

    run_data["metric_results"]["value"].append(metric_id)


def build_yaml_evaluation(yaml_data):
    metrics = yaml_data.get("metrics")

    if not isinstance(metrics, dict):
        return {}

    model_variant = yaml_data.get("model_key") or yaml_data.get("model_name")
    evidence = "YAML fields train_task, test_task, targets and metrics"

    evaluation = {
        "evaluation_runs": {},
        "benchmark_releases": {
            "matbench_discovery_benchmark_release": {
                "type": evidence_field(
                    "BenchmarkRelease",
                    "ontology/evaluationonto.ttl: BenchmarkRelease",
                ),
                "tasks": evidence_field([], evidence),
            }
        },
        "tasks": {},
        "metric_results": {},
    }

    task_ids = set()
    discovery_metrics = metrics.get("discovery")

    if isinstance(discovery_metrics, dict):
        classification_run_id = scoped_evaluation_id(
            model_variant,
            "matbench_discovery_thermodynamic_stability_classification"
        )
        regression_run_id = scoped_evaluation_id(
            model_variant,
            "matbench_discovery_convex_hull_distance_regression"
        )

        init_evaluation_run(
            evaluation,
            classification_run_id,
            "thermodynamic_stability_classification_task",
            model_variant,
            "YAML metrics.discovery",
        )
        init_evaluation_run(
            evaluation,
            regression_run_id,
            "convex_hull_distance_regression_task",
            model_variant,
            "YAML metrics.discovery",
        )

        for split_path, metric_name, metric_value in iter_metric_leaf_values(
            discovery_metrics
        ):
            if metric_name in CLASSIFICATION_METRICS:
                add_yaml_metric_result(
                    evaluation,
                    classification_run_id,
                    "discovery",
                    split_path,
                    metric_name,
                    metric_value,
                )
                task_ids.add("thermodynamic_stability_classification_task")

            if metric_name in CONVEX_HULL_REGRESSION_METRICS:
                add_yaml_metric_result(
                    evaluation,
                    regression_run_id,
                    "discovery",
                    split_path,
                    metric_name,
                    metric_value,
                )
                task_ids.add("convex_hull_distance_regression_task")

        for run_id in (classification_run_id, regression_run_id):
            if not evaluation["evaluation_runs"][run_id]["metric_results"]["value"]:
                del evaluation["evaluation_runs"][run_id]

    phonon_metrics = metrics.get("phonons")

    if isinstance(phonon_metrics, dict):
        run_id = scoped_evaluation_id(
            model_variant,
            "matbench_discovery_phonon_thermal_conductivity_contribution"
        )
        task_id = "phonon_thermal_conductivity_contribution_task"
        init_evaluation_run(
            evaluation,
            run_id,
            task_id,
            model_variant,
            "YAML metrics.phonons",
        )

        for split_path, metric_name, metric_value in iter_metric_leaf_values(
            phonon_metrics
        ):
            add_yaml_metric_result(
                evaluation,
                run_id,
                "phonons",
                split_path,
                metric_name,
                metric_value,
            )
            task_ids.add(task_id)

        if not evaluation["evaluation_runs"][run_id]["metric_results"]["value"]:
            del evaluation["evaluation_runs"][run_id]

    geo_opt_metrics = metrics.get("geo_opt")

    if isinstance(geo_opt_metrics, dict):
        run_id = scoped_evaluation_id(
            model_variant,
            "matbench_discovery_relaxed_structure_matching_rmsd"
        )
        task_id = "relaxed_structure_matching_rmsd_task"
        init_evaluation_run(
            evaluation,
            run_id,
            task_id,
            model_variant,
            "YAML metrics.geo_opt",
        )

        for split_path, metric_name, metric_value in iter_metric_leaf_values(
            geo_opt_metrics
        ):
            add_yaml_metric_result(
                evaluation,
                run_id,
                "geo_opt",
                split_path,
                metric_name,
                metric_value,
            )
            task_ids.add(task_id)

        if not evaluation["evaluation_runs"][run_id]["metric_results"]["value"]:
            del evaluation["evaluation_runs"][run_id]

    for task_id in sorted(task_ids):
        task_info = MATBENCH_TASKS[task_id]
        evaluation["tasks"][task_id] = {
            "type": evidence_field(task_info["type"], "ontology/evaluation_individuals.ttl"),
            "dataset": evidence_field(task_info["dataset"], evidence),
        }

    evaluation["benchmark_releases"]["matbench_discovery_benchmark_release"][
        "tasks"
    ] = evidence_field(sorted(task_ids), evidence)

    if not evaluation["evaluation_runs"]:
        return {}

    return evaluation


def load_yaml_evaluation_from_sources(data):
    sources = data.get("_sources", {})
    yaml_file = sources.get("yaml_file")

    if not yaml_file:
        return {}

    yaml_path = Path(yaml_file)

    if not yaml_path.exists():
        return {}

    with yaml_path.open("r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file) or {}

    return build_yaml_evaluation(yaml_data)


def load_yaml_data_from_sources(data):
    sources = data.get("_sources", {})
    yaml_file = sources.get("yaml_file")

    if not yaml_file:
        return {}

    yaml_path = Path(yaml_file)

    if not yaml_path.exists():
        return {}

    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def yaml_model_variant_field(yaml_data):
    model_key = yaml_data.get("model_key")

    if model_key:
        return evidence_field(model_key, "YAML model_key")

    model_name = yaml_data.get("model_name")

    if model_name:
        return evidence_field(model_name, "YAML model_name")

    return None


def yaml_scalar_hyperparameters(yaml_data):
    hyperparams = yaml_data.get("hyperparams")

    if not isinstance(hyperparams, dict):
        return {}

    scalars = {}

    for name, value in hyperparams.items():
        if isinstance(value, (dict, list, tuple, set)):
            continue
        scalars[str(name)] = value

    return scalars


def merge_yaml_training_and_hyperparameters(data):
    sources = data.get("_sources", {})
    yaml_file = sources.get("yaml_file")

    if not yaml_file:
        return data

    yaml_path = Path(yaml_file)

    if not yaml_path.exists():
        return data

    with yaml_path.open("r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file) or {}

    hyperparams = yaml_scalar_hyperparameters(yaml_data)

    if not hyperparams:
        return data

    enriched = dict(data)
    raw_architectures = enriched.get("architectures", {})

    if isinstance(raw_architectures, dict):
        architectures = dict(raw_architectures)
    elif isinstance(raw_architectures, list):
        architectures = {
            get_value(entry.get("name")) or get_value(entry.get("id")): entry
            for entry in raw_architectures
            if isinstance(entry, dict)
            and (get_value(entry.get("name")) or get_value(entry.get("id")))
        }
    else:
        architectures = {}

    raw_training = enriched.get("training", {})
    if isinstance(raw_training, dict):
        training = dict(raw_training)
    elif isinstance(raw_training, list):
        training = {
            "training_notes": evidence_field(
                [get_value(item) for item in raw_training],
                "LLM training list"
            )
        }
    elif has_meaningful_value(raw_training):
        training = {
            "training_notes": evidence_field(
                get_value(raw_training),
                "LLM training value"
            )
        }
    else:
        training = {}
    training_runs = list(training.get("training_runs", []))

    # Reuse YAML hyperparameters as generic ML hyperparameters when the extraction
    # missed them, so the converter can still emit value individuals.
    for architecture_name, architecture in list(architectures.items()):
        if not isinstance(architecture, dict):
            continue

        merged_architecture = dict(architecture)
        existing_hyperparameters = normalize_list(merged_architecture.get("hyperparameters"))
        existing_names = {slugify(item.get("value") if isinstance(item, dict) else item) for item in existing_hyperparameters}

        merged_hyperparameters = list(existing_hyperparameters)

        for hp_name, hp_value in hyperparams.items():
            hp_slug = slugify(hp_name)

            if hp_slug not in existing_names:
                merged_hyperparameters.append(
                    evidence_field(hp_name, f"YAML hyperparams.{hp_name}")
                )
                existing_names.add(hp_slug)

            if not has_meaningful_value(merged_architecture.get(hp_name)):
                merged_architecture[hp_name] = evidence_field(
                    hp_value,
                    f"YAML hyperparams.{hp_name}"
                )

        merged_architecture["hyperparameters"] = merged_hyperparameters
        architectures[architecture_name] = merged_architecture

    if architectures and not any(is_training_run_data(name, data) for name, data in iter_training_runs(training)):
        training_runs.append({
            "id": evidence_field(
                f"{slugify(yaml_data.get('model_key') or yaml_data.get('model_name') or 'model')}_training_run",
                "YAML model_key/model_name"
            ),
            "type": evidence_field(
                "MachineLearningTrainingRun",
                "ontology/trainingonto.ttl: MachineLearningTrainingRun"
            ),
            "architectures": evidence_field(
                list(architectures.keys()),
                "YAML hyperparams"
            ),
        })

    # Populate run-level training fields directly from YAML when missing.
    for index, run_data in enumerate(training_runs):
        if not isinstance(run_data, dict):
            continue

        merged_run = dict(run_data)

        if not has_meaningful_value(merged_run.get("optimizer")) and "optimizer" in hyperparams:
            merged_run["optimizer"] = evidence_field(
                hyperparams["optimizer"],
                "YAML hyperparams.optimizer"
            )

        if not has_meaningful_value(merged_run.get("loss_function")) and "loss" in hyperparams:
            merged_run["loss_function"] = evidence_field(
                hyperparams["loss"],
                "YAML hyperparams.loss"
            )

        training_runs[index] = merged_run

    training["training_runs"] = training_runs
    enriched["training"] = training
    enriched["architectures"] = architectures
    return enriched


def merge_evaluation(json_evaluation, yaml_evaluation):
    if not yaml_evaluation:
        return json_evaluation

    if not isinstance(json_evaluation, dict):
        return yaml_evaluation

    merged = dict(json_evaluation)
    stale_metric_ids = set()

    evaluation_runs = merged.get("evaluation_runs")
    if isinstance(evaluation_runs, dict):
        filtered_runs = {}

        for run_id, run_data in evaluation_runs.items():
            if slugify(run_id).startswith("matbench_discovery_"):
                if isinstance(run_data, dict):
                    stale_metric_ids.update(
                        slugify(get_value(metric))
                        for metric in normalize_list(run_data.get("metric_results"))
                        if get_value(metric)
                    )
                continue

            filtered_runs[run_id] = run_data

        merged["evaluation_runs"] = filtered_runs

    metric_results = merged.get("metric_results")
    if isinstance(metric_results, dict) and stale_metric_ids:
        merged["metric_results"] = {
            metric_id: metric_data
            for metric_id, metric_data in metric_results.items()
            if slugify(metric_id) not in stale_metric_ids
        }

    for key, value in yaml_evaluation.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {
                **merged[key],
                **value,
            }
        else:
            merged[key] = value

    return merged


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

    evidence = get_evidence(evidence_source)
    if evidence is None and not isinstance(evidence_source, dict):
        evidence = evidence_source

    if not evidence:
        return

    escaped = str(evidence).replace("\\", "\\\\").replace('"', '\\"')

    line = (
        f"<< {subject.n3()} {predicate.n3()} {obj.n3()} >> "
        f'prov:wasDerivedFrom "{escaped}"'
    )

    source_url = get_source(evidence_source) or pdf_url

    if source_url:
        safe_source_url = str(source_url).replace("\\", "\\\\").replace('"', '\\"')
        line += f" ; dcterms:source <{safe_source_url}>"

    line += " ."

    rdf_star_lines.append(line)


def iter_architectures(architectures):
    if isinstance(architectures, list):
        for entry in architectures:
            if not isinstance(entry, dict):
                continue
            name = get_value(entry.get("name")) or get_value(entry.get("id"))
            if name:
                yield name, entry
        return

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
        elif has_meaningful_value(data):
            yield name, {
                "description": evidence_field(
                    get_value(data),
                    f"LLM architectures.{name}"
                )
            }


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

    for item in normalize_list(hyperparameters_field):
        item_data = item if isinstance(item, dict) else None
        if item_data:
            name = (
                get_value(item_data.get("name"))
                or get_value(item_data.get("parameter"))
                or get_value(item_data.get("hyperparameter"))
                or get_value(item_data.get("label"))
            )
        else:
            name = get_value(item)
        if not name:
            continue
        name = str(name)
        data = architecture.get(name)
        if not isinstance(data, dict):
            if item_data:
                data = item_data
            else:
                evidence = get_evidence(hyperparameters_field)
                data = {"evidence": evidence} if evidence else {}
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
            if is_training_run_data(name, data):
                yield name, data
        return

    if isinstance(training.get("training_run_details"), dict):
        for name, data in training["training_run_details"].items():
            if is_training_run_data(name, data):
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

        if name and is_training_run_data(name, data):
            yielded_names.add(name)
            yield name, data

    for name, data in training.items():
        if name in {"training_runs", "training_run_loop", "training_run_details"}:
            continue
        if name in yielded_names:
            continue
        if is_training_run_data(name, data):
            yield name, data


def run_architecture_names(run_data, architecture_entries):
    requested_names = []

    for architecture_name in normalize_list(run_data.get("architectures")):
        architecture_name = get_value(architecture_name)

        if architecture_name:
            requested_names.append(architecture_name)

    if not requested_names and len(architecture_entries) == 1:
        return [architecture_entries[0][0]]

    architecture_by_slug = {
        slugify(architecture_name): architecture_name
        for architecture_name, _ in architecture_entries
    }

    matched_names = []

    for architecture_name in requested_names:
        matched_name = architecture_by_slug.get(slugify(architecture_name))

        if matched_name:
            matched_names.append(matched_name)
        else:
            matched_names.append(architecture_name)

    return matched_names


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

    for loop_key in ("benchmark_task_loop", "task_loop"):
        if isinstance(evaluation.get(loop_key), dict):
            for name, data in evaluation[loop_key].items():
                if isinstance(data, dict):
                    yield name, data
            return

    yielded_names = set()

    task_entries = evaluation.get("benchmark_tasks", evaluation.get("tasks", []))

    if isinstance(task_entries, dict):
        for name, data in task_entries.items():
            if isinstance(data, dict):
                yielded_names.add(name)
                yield name, data
        return

    for entry in task_entries:
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
        if name in {
            "benchmark_tasks",
            "benchmark_task_loop",
            "tasks",
            "task_loop",
            "evaluation_runs",
            "evaluation_run_loop",
            "benchmark_releases",
            "benchmark_release_loop",
            "folds",
            "fold_loop",
            "metric_results",
            "metric_result_loop",
            "metric_types",
            "metric_type_loop",
            "benchmark_suite",
            "benchmark_release",
        }:
            continue
        if name in yielded_names:
            continue
        if isinstance(data, dict):
            yield name, data


def iter_named_entities(section, list_key, loop_key, skip_keys=()):
    if not isinstance(section, dict):
        return

    if isinstance(section.get(loop_key), dict):
        for name, data in section[loop_key].items():
            if isinstance(data, dict):
                yield name, data
        return

    yielded_names = set()
    entries = section.get(list_key, [])

    if isinstance(entries, dict):
        for name, data in entries.items():
            if isinstance(data, dict):
                yielded_names.add(name)
                yield name, data
        return

    for entry in normalize_list(entries):
        if isinstance(entry, dict):
            name = get_value(entry.get("name")) or get_value(entry.get("id"))
            data = entry
        else:
            name = get_value(entry)
            data = section.get(name)

        if name and isinstance(data, dict):
            yielded_names.add(name)
            yield name, data

    skipped = {list_key, loop_key, *skip_keys}

    for name, data in section.items():
        if name in skipped:
            continue
        if name in yielded_names:
            continue
        if isinstance(data, dict):
            yield name, data


def iter_evaluation_runs(evaluation):
    yield from iter_named_entities(
        evaluation,
        "evaluation_runs",
        "evaluation_run_loop",
        skip_keys={
            "benchmark_releases",
            "benchmark_release_loop",
            "benchmark_tasks",
            "benchmark_task_loop",
            "tasks",
            "task_loop",
            "folds",
            "fold_loop",
            "metric_results",
            "metric_result_loop",
            "metric_types",
            "metric_type_loop",
            "benchmark_suite",
            "benchmark_release",
        },
    )


def iter_benchmark_releases(evaluation):
    yield from iter_named_entities(
        evaluation,
        "benchmark_releases",
        "benchmark_release_loop",
        skip_keys={
            "evaluation_runs",
            "evaluation_run_loop",
            "benchmark_tasks",
            "benchmark_task_loop",
            "tasks",
            "task_loop",
            "folds",
            "fold_loop",
            "metric_results",
            "metric_result_loop",
            "metric_types",
            "metric_type_loop",
        },
    )


def named_data(section, key, name):
    if not isinstance(section, dict) or not name:
        return {}

    direct = section.get(name)
    if isinstance(direct, dict):
        return direct

    values = section.get(key)
    if isinstance(values, dict) and isinstance(values.get(name), dict):
        return values[name]

    loop = section.get(f"{key[:-1]}_loop")
    if isinstance(loop, dict) and isinstance(loop.get(name), dict):
        return loop[name]

    return {}


def add_dataset_reference(graph, dataset_name, dataset_class=DATA.Dataset):
    dataset_name = get_value(dataset_name)

    if not dataset_name:
        return None

    dataset_uri = DATAIND[slugify(dataset_name)]
    add_named_individual(graph, dataset_uri, dataset_class, dataset_name)
    return dataset_uri


def add_dataset_split_defaults(graph, split_uri, split_name):
    split_slug = slugify(split_name)
    representation_uri = DATAIND[f"{split_slug}_data_representation"]

    add_named_individual(
        graph,
        representation_uri,
        DATA.DataRepresentation,
        f"{split_name} data representation"
    )
    graph.add((split_uri, DATA.hasDataRepresentation, representation_uri))

    parent_name = (
        "kappa-103 phonon benchmark set"
        if split_slug == "kappa_103"
        else "WBM test set"
    )
    parent_uri = add_dataset_reference(graph, parent_name, DATA.Dataset)
    graph.add((split_uri, DATA.wasDerivedFromDataset, parent_uri))


def add_model_materialization(graph, name, rdf_class=TRAIN.ModelMaterialization):
    name = get_value(name)

    if not name:
        return None

    materialization_uri = TRIND[f"{slugify(name)}_materialization"]
    add_named_individual(graph, materialization_uri, rdf_class, name)
    return materialization_uri


def add_model_variant_reference(graph, name):
    name = get_value(name)

    if not name:
        return None

    variant_uri = ARCHIND[f"{slugify(name)}_variant"]
    add_named_individual(graph, variant_uri, ARCH.ModelVariant, name)
    return variant_uri


def add_evaluation_reference(graph, name, rdf_class, suffix):
    name = get_value(name)

    if not name:
        return None

    name_slug = slugify(name)

    if name_slug.endswith(f"_{suffix}"):
        uri = EVALIND[name_slug]
    else:
        uri = EVALIND[f"{name_slug}_{suffix}"]

    add_named_individual(graph, uri, rdf_class, name)
    return uri


def canonical_task_id(task_name):
    task_name = get_value(task_name)

    if not task_name:
        return None

    return TASK_ALIASES.get(slugify(task_name), slugify(task_name))


def add_task_reference(graph, task_name, rdf_class=EVAL.Task):
    task_id = canonical_task_id(task_name)

    if not task_id:
        return None

    task_uri = EVALIND[task_id]
    label = TASK_LABELS.get(task_id, get_value(task_name))
    task_class = TASK_CLASSES.get(task_id, rdf_class)

    add_named_individual(graph, task_uri, EVAL.Task, label)

    if task_class != EVAL.Task:
        graph.add((task_uri, RDF.type, task_class))

    return task_uri


def add_algorithm_reference(graph, name, rdf_class):
    name = get_value(name)

    if not name:
        return None

    uri = TRIND[slugify(name)]
    add_named_individual(graph, uri, rdf_class, name)
    return uri


def build_graph(data):
    data = merge_yaml_training_and_hyperparameters(data)
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
    yaml_data = load_yaml_data_from_sources(data)

    raw_model = data.get("model", {})
    if isinstance(raw_model, dict):
        model = raw_model
    elif has_meaningful_value(raw_model):
        model = {
            "family": evidence_field(get_value(raw_model), "LLM model value"),
            "variant": yaml_model_variant_field(yaml_data) or evidence_field(
                get_value(raw_model),
                "LLM model value"
            ),
        }
    else:
        model = {}
    architectures = data.get("architectures", {})
    training = data.get("training", {})
    datasets = data.get("datasets") or {"datasets": training.get("datasets", [])}
    evaluation = merge_evaluation(
        data.get("evaluation", {}),
        load_yaml_evaluation_from_sources(data),
    )

    family_field = model.get("family")
    variant_field = yaml_model_variant_field(yaml_data) or model.get("variant")
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
        graph.add((variant_uri, PROV.wasDerivedFrom, URIRef(pdf_url)))

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
    architecture_hyperparameters = {}
    architecture_entries = list(iter_architectures(architectures))
    all_hyperparameters = []

    for architecture_name, architecture in architecture_entries:
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
        architecture_hyperparameters[architecture_name] = []

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

        if architecture_name == architecture_entries[0][0]:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                variant_uri,
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

            hp_class = class_or_default(
                TRAIN,
                hp_data.get("ontology_class"),
                TRAIN.MachineLearningHyperparameter
            )

            add_named_individual(graph, hp_uri, hp_class, hp_name)
            graph.add((hp_uri, RDF.type, TRAIN.MachineLearningHyperparameter))
            hp_entry = (hp_name, hp_data)
            architecture_hyperparameters[architecture_name].append(hp_entry)
            all_hyperparameters.append(hp_entry)

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

        run_hyperparameters = []
        seen_hyperparameter_slugs = set()

        for architecture_name in run_architecture_names(run_data, architecture_entries):
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

            for hp_name, hp_data in architecture_hyperparameters.get(architecture_name, []):
                hp_slug = slugify(hp_name)

                if hp_slug in seen_hyperparameter_slugs:
                    continue

                seen_hyperparameter_slugs.add(hp_slug)
                run_hyperparameters.append((hp_name, hp_data))

        for hp_name, hp_data in all_hyperparameters:
            hp_slug = slugify(hp_name)

            if hp_slug in seen_hyperparameter_slugs:
                continue

            seen_hyperparameter_slugs.add(hp_slug)
            run_hyperparameters.append((hp_name, hp_data))

        for hp_name, hp_data in run_hyperparameters:
            hp_slug = slugify(hp_name)
            hp_uri = TRIND[hp_slug]
            hp_value_uri = TRIND[f"{run_slug}_{hp_slug}_value"]

            add_named_individual(
                graph,
                hp_value_uri,
                TRAIN.MachineLearningHyperparameterValue,
                f"{run_name} {hp_name} value"
            )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                hp_value_uri,
                TRAIN.ofHyperparameter,
                hp_uri,
                hp_data.get("ontology_class") or hp_data,
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
            graph.add((run_uri, RDF.type, TRAIN.MachineLearningAdaptationRun))
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
            materialization_name = get_value(materialization_name)

            if not materialization_name:
                continue

            materialization_uri = TRIND[
                f"{run_slug}_{slugify(materialization_name)}_checkpoint"
            ]
            add_named_individual(
                graph,
                materialization_uri,
                TRAIN.Checkpoint,
                materialization_name
            )

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

        if labelling_method or normalize_list(dataset_data.get("target_features")):
            graph.add((dataset_uri, RDF.type, DATA.LabeledDataset))

        if normalize_list(dataset_data.get("input_features")):
            graph.add((dataset_uri, RDF.type, TRAIN.TrainingDataset))

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

    if benchmark_suite:
        add_evaluation_reference(
            graph,
            benchmark_suite,
            EVAL.BenchmarkSuite,
            "benchmark_suite"
        )

    benchmark_release_uris = {}

    for release_name, release_data in iter_benchmark_releases(evaluation):
        release_uri = add_evaluation_reference(
            graph,
            release_name,
            EVAL.BenchmarkRelease,
            "benchmark_release"
        )

        if release_uri is None:
            continue

        benchmark_release_uris[slugify(release_name)] = release_uri

        for task_name in normalize_list(release_data.get("tasks")):
            task_uri = add_task_reference(graph, task_name, EVAL.Task)

            if task_uri is None:
                continue

            add_triple_with_evidence(
                graph, rdf_star_lines,
                release_uri,
                EVAL.hasTask,
                task_uri,
                release_data.get("tasks"),
                pdf_url
            )

    benchmark_release_field = evaluation.get("benchmark_release")
    benchmark_release = get_value(benchmark_release_field)

    if benchmark_release:
        release_uri = add_evaluation_reference(
            graph,
            benchmark_release,
            EVAL.BenchmarkRelease,
            "benchmark_release"
        )
        benchmark_release_uris[slugify(benchmark_release)] = release_uri

    task_uris = {}

    for task_name, task_data in iter_benchmark_tasks(evaluation):
        task_class = class_or_default(EVAL, task_data.get("type"), EVAL.Task)

        task_uri = add_task_reference(graph, task_name, task_class)

        if task_uri is None:
            continue

        task_uris[slugify(task_name)] = task_uri
        task_uris[canonical_task_id(task_name)] = task_uri

        dataset_uri = add_dataset_reference(graph, task_data.get("dataset"))

        if dataset_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                task_uri,
                EVAL.usesDataset,
                dataset_uri,
                task_data.get("dataset"),
                pdf_url
            )

    fold_uris = {}

    for fold_name, fold_data in iter_named_entities(
        evaluation,
        "folds",
        "fold_loop",
        skip_keys={
            "evaluation_runs",
            "evaluation_run_loop",
            "benchmark_releases",
            "benchmark_release_loop",
            "benchmark_tasks",
            "benchmark_task_loop",
            "tasks",
            "task_loop",
            "metric_results",
            "metric_result_loop",
            "metric_types",
            "metric_type_loop",
        },
    ):
        fold_uri = add_evaluation_reference(graph, fold_name, EVAL.Fold, "fold")

        if fold_uri is None:
            continue

        fold_uris[slugify(fold_name)] = fold_uri

        index_field = fold_data.get("index")
        index = get_value(index_field)

        if index is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                fold_uri,
                EVAL.hasIndex,
                literal_from_value(index),
                index_field,
                pdf_url
            )

        for split_key, predicate in (
            ("train_split", EVAL.hasTrainSplit),
            ("validation_split", EVAL.hasValidationSplit),
            ("test_split", EVAL.hasTestSplit),
        ):
            split_field = fold_data.get(split_key)
            split_uri = add_dataset_reference(
                graph,
                split_field,
                DATA.DatasetSplit
            )

            if split_uri is None:
                continue

            add_dataset_split_defaults(graph, split_uri, get_value(split_field))

            add_triple_with_evidence(
                graph, rdf_star_lines,
                fold_uri,
                predicate,
                split_uri,
                split_field,
                pdf_url
            )

    metric_result_uris = {}

    for metric_name, metric_data in iter_named_entities(
        evaluation,
        "metric_results",
        "metric_result_loop",
        skip_keys={
            "evaluation_runs",
            "evaluation_run_loop",
            "benchmark_releases",
            "benchmark_release_loop",
            "benchmark_tasks",
            "benchmark_task_loop",
            "tasks",
            "task_loop",
            "folds",
            "fold_loop",
            "metric_types",
            "metric_type_loop",
        },
    ):
        metric_type_field = metric_data.get("metric_type")
        metric_type = get_value(metric_type_field) or metric_name
        metric_value_field = metric_data.get("metric_value")
        metric_value = get_value(metric_value_field)
        unit = get_value(metric_data.get("unit"))
        metric_label = " ".join(
            str(part)
            for part in (metric_type, metric_value, unit)
            if part is not None
        )

        metric_uri = add_evaluation_reference(
            graph,
            metric_name,
            EVAL.MetricResult,
            "metric_result"
        )

        if metric_uri is None:
            continue

        if metric_label:
            graph.set((metric_uri, RDFS.label, Literal(metric_label, datatype=XSD.string)))

        metric_result_uris[slugify(metric_name)] = metric_uri

        metric_type_uri = add_evaluation_reference(
            graph,
            metric_type,
            EVAL.MetricType,
            "metric_type"
        )

        if metric_type_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                metric_uri,
                EVAL.hasMetricType,
                metric_type_uri,
                metric_type_field,
                pdf_url
            )

        if metric_value is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                metric_uri,
                EVAL.hasMetricValue,
                literal_from_value(metric_value),
                metric_value_field,
                pdf_url
            )

    for run_name, run_data in iter_evaluation_runs(evaluation):
        run_uri = add_evaluation_reference(
            graph,
            run_name,
            EVAL.MachineLearningEvaluationRun,
            "evaluation_run"
        )

        if run_uri is None:
            continue

        run_variant_uri = add_model_variant_reference(
            graph,
            run_data.get("model_variant") or variant_name
        )

        if run_variant_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                EVAL.evaluatesModelVariant,
                run_variant_uri,
                run_data.get("model_variant") or variant_field,
                pdf_url
            )

        release_field = run_data.get("benchmark_release")
        release = get_value(release_field)
        release_uri = benchmark_release_uris.get(slugify(release))

        if release and release_uri is None:
            release_uri = add_evaluation_reference(
                graph,
                release,
                EVAL.BenchmarkRelease,
                "benchmark_release"
            )
            benchmark_release_uris[slugify(release)] = release_uri

        if release_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                EVAL.usesBenchmarkRelease,
                release_uri,
                release_field,
                pdf_url
            )

        task_field = run_data.get("task")
        task = get_value(task_field)
        task_uri = task_uris.get(slugify(task)) or task_uris.get(canonical_task_id(task))

        if task and task_uri is None:
            task_uri = add_task_reference(graph, task, EVAL.Task)
            task_uris[slugify(task)] = task_uri
            task_uris[canonical_task_id(task)] = task_uri

        if task_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                run_uri,
                EVAL.evaluatesTask,
                task_uri,
                task_field,
                pdf_url
            )

        dataset_uri = add_dataset_reference(graph, run_data.get("dataset"))

        if dataset_uri is not None and task_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                task_uri,
                EVAL.usesDataset,
                dataset_uri,
                run_data.get("dataset"),
                pdf_url
            )

        result_uri = EVALIND[f"{slugify(run_name)}_benchmark_result"]
        add_named_individual(
            graph,
            result_uri,
            EVAL.BenchmarkResult,
            f"{run_name} benchmark result"
        )

        add_triple_with_evidence(
            graph, rdf_star_lines,
            run_uri,
            EVAL.producesBenchmarkResult,
            result_uri,
            run_data.get("type"),
            pdf_url
        )

        if task_uri is not None:
            add_triple_with_evidence(
                graph, rdf_star_lines,
                result_uri,
                EVAL.hasResultTask,
                task_uri,
                task_field,
                pdf_url
            )

        for fold_index, fold_entry in enumerate(normalize_list(run_data.get("folds"))):
            fold_name = get_value(fold_entry)

            if not fold_name:
                continue

            fold_uri = fold_uris.get(slugify(fold_name))

            if fold_uri is None:
                fold_uri = add_evaluation_reference(
                    graph,
                    fold_name,
                    EVAL.Fold,
                    "fold"
                )
                fold_uris[slugify(fold_name)] = fold_uri
                graph.add((
                    fold_uri,
                    EVAL.hasIndex,
                    Literal(fold_index, datatype=XSD.integer)
                ))

        for metric_entry in normalize_list(run_data.get("metric_results")):
            metric_name = get_value(metric_entry)

            if not metric_name:
                continue

            metric_data = named_data(evaluation, "metric_results", metric_name)
            metric_uri = metric_result_uris.get(slugify(metric_name))

            if metric_uri is None:
                metric_type = get_value(metric_data.get("metric_type")) or metric_name
                metric_uri = add_evaluation_reference(
                    graph,
                    metric_name,
                    EVAL.MetricResult,
                    "metric_result"
                )
                metric_result_uris[slugify(metric_name)] = metric_uri

                metric_type_uri = add_evaluation_reference(
                    graph,
                    metric_type,
                    EVAL.MetricType,
                    "metric_type"
                )

                if metric_type_uri is not None:
                    add_triple_with_evidence(
                        graph, rdf_star_lines,
                        metric_uri,
                        EVAL.hasMetricType,
                        metric_type_uri,
                        metric_data.get("metric_type") or metric_entry,
                        pdf_url
                    )

                metric_value_field = metric_data.get("metric_value")
                metric_value = get_value(metric_value_field)

                if metric_value is not None:
                    add_triple_with_evidence(
                        graph, rdf_star_lines,
                        metric_uri,
                        EVAL.hasMetricValue,
                        literal_from_value(metric_value),
                        metric_value_field,
                        pdf_url
                    )

            add_triple_with_evidence(
                graph, rdf_star_lines,
                result_uri,
                EVAL.hasMetricResult,
                metric_uri,
                metric_entry,
                pdf_url
            )

            for fold_name in normalize_list(run_data.get("folds")):
                fold_uri = fold_uris.get(slugify(fold_name))

                if fold_uri is None:
                    continue

                add_triple_with_evidence(
                    graph, rdf_star_lines,
                    fold_uri,
                    EVAL.producesMetricResult,
                    metric_uri,
                    metric_entry,
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
