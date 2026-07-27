import argparse
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, OWL, XSD


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "ttl_datasets" / "matbench_site_context_generated.ttl"
DEFAULT_LLM_JSON = BASE_DIR / "outputs" / "json" / "matbench_site_context_llm_extraction.json"
RAW_LLM_DIR = BASE_DIR / "outputs" / "raw_llm"
SITE_URL = "https://matbench-discovery.materialsproject.org"
DATASETS_URL = f"{SITE_URL}/data/sets"

TASK_URLS = {
    "diatomics": f"{SITE_URL}/tasks/diatomics",
    "discovery": f"{SITE_URL}/tasks/discovery",
    "geo_opt": f"{SITE_URL}/tasks/geo-opt",
    "md": f"{SITE_URL}/tasks/md",
    "phonons": f"{SITE_URL}/tasks/phonons",
}

DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")
DATAIND = Namespace("https://k.loria.fr/ontologies/dataset-individuals#")
EVAL = Namespace("https://k.loria.fr/ontologies/evaluationonto#")
EVALIND = Namespace("https://k.loria.fr/ontologies/evaluationonto-individuals#")

load_dotenv(BASE_DIR / ".env")


TASK_SPECS = {
    "diatomics": {
        "id": "diatomics_task",
        "label": "Diatomics",
        "class": EVAL.RegressionTask,
        "datasets": ["diatomic_reference_curves"],
    },
    "discovery": {
        "id": "matbench_discovery_discovery_task",
        "label": "Discovery",
        "class": EVAL.ClassificationTask,
        "datasets": ["wbm_test_set", "materials_project_reference_hull"],
    },
    "geo_opt": {
        "id": "geometry_optimization_task",
        "label": "Geometry optimization",
        "class": EVAL.RegressionTask,
        "datasets": ["wbm_test_set"],
    },
    "md": {
        "id": "molecular_dynamics_task",
        "label": "Molecular dynamics",
        "class": EVAL.RegressionTask,
        "datasets": ["md_benchmark_set"],
    },
    "phonons": {
        "id": "phonons_task",
        "label": "Phonons",
        "class": EVAL.RegressionTask,
        "datasets": ["kappa_103_phonon_benchmark_set", "mdr_mp_pbe_omega_q"],
    },
}


SUPPORT_DATASETS = {
    "diatomic_reference_curves": "Diatomic DFT reference curves",
    "materials_project_reference_hull": "Materials Project reference convex hull",
    "md_benchmark_set": "Molecular dynamics benchmark set",
    "kappa_103_phonon_benchmark_set": "kappa-103 phonon benchmark set",
    "wbm_test_set": "WBM test set",
    "mdr_mp_pbe_omega_q": "MDR-MP PBE omega_q",
}


def slugify(value):
    value = str(value).strip().lower()
    value = value.replace("ω", "omega")
    chars = [char if char.isalnum() else "_" for char in value]
    return "_".join(part for part in "".join(chars).split("_") if part) or "unnamed"


def configured_openrouter_model():
    return os.getenv("OPENROUTER_MODEL", "meituan/longcat-2.0")


def fetch_html(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    return response.text


def cell_text(cell):
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()


def header_text(cell):
    text = cell_text(cell)
    return text.replace("↑", "").replace("↓", "").strip()


def cell_number(cell):
    value = cell.get("data-sort-value")
    if not value:
        text = cell_text(cell).lower()
        if text in {"", "n/a"}:
            return None
        value = text
    try:
        return int(float(value))
    except ValueError:
        return None


def cell_bool(cell):
    svg = cell.find("svg")
    style = (svg.get("style", "") if svg else "") + " " + cell.get("style", "")
    if "lightgreen" in style:
        return True
    if "lightcoral" in style:
        return False
    return None


def parse_created(cell):
    span = cell.find("span")
    value = cell_text(cell)
    title = span.get("title") if span else None
    for candidate in (title, value):
        if not candidate:
            continue
        candidate = candidate.split(",", 1)[-1].strip() if "," in candidate else candidate
        for fmt in ("%b %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                pass
    return value or None


def parse_dataset_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    headers = [header_text(th) for th in table.select("thead th")]
    rows = []

    for tr in table.select("tbody tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue

        by_header = {headers[index]: cell for index, cell in enumerate(cells) if index < len(headers)}
        name_cell = by_header.get("Name")
        if name_cell is None:
            continue

        link = name_cell.find("a")
        label = cell_text(link or name_cell)
        if not label:
            continue

        href = link.get("href") if link else None
        title = link.get("title") if link else None

        rows.append({
            "id": slugify(href.rsplit("/", 1)[-1] if href else label),
            "label": label,
            "title": title,
            "page": urljoin(SITE_URL, href) if href else DATASETS_URL,
            "structures": cell_number(by_header["Structures"]) if "Structures" in by_header else None,
            "materials": cell_number(by_header["Materials"]) if "Materials" in by_header else None,
            "created": parse_created(by_header["Created"]) if "Created" in by_header else None,
            "open": cell_bool(by_header["Open"]) if "Open" in by_header else None,
            "static": cell_bool(by_header["Static"]) if "Static" in by_header else None,
            "license": cell_text(by_header["License"]) if "License" in by_header else None,
            "method": (by_header["Method"].find("span") or by_header["Method"]).get("title") if "Method" in by_header else None,
            "api_links": [urljoin(SITE_URL, a["href"]) for a in by_header.get("API", BeautifulSoup("", "html.parser")).find_all("a", href=True)],
            "links": [urljoin(SITE_URL, a["href"]) for a in by_header.get("Links", BeautifulSoup("", "html.parser")).find_all("a", href=True)],
        })

    return rows


def extract_task_context(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        return "", ""

    title = cell_text(main.find("h1") or "")
    paragraphs = []
    for p in main.find_all("p"):
        text = cell_text(p)
        if text:
            paragraphs.append(text)
        if len(" ".join(paragraphs)) > 1800:
            break

    return title, " ".join(paragraphs)


def collect_scraped_context():
    dataset_rows = parse_dataset_rows(fetch_html(DATASETS_URL))
    task_contexts = {}

    for task_key, task_url in TASK_URLS.items():
        page_title, description = extract_task_context(fetch_html(task_url))
        task_contexts[task_key] = {
            "url": task_url,
            "title": page_title or TASK_SPECS[task_key]["label"],
            "description": description,
            "seed_task_id": TASK_SPECS[task_key]["id"],
            "seed_task_type": (
                "ClassificationTask"
                if TASK_SPECS[task_key]["class"] == EVAL.ClassificationTask
                else "RegressionTask"
            ),
            "seed_dataset_ids": TASK_SPECS[task_key]["datasets"],
        }

    return dataset_rows, task_contexts


def compact_dataset_for_llm(dataset):
    return {
        "id": dataset["id"],
        "label": dataset["label"],
        "title": dataset.get("title"),
        "page": dataset.get("page"),
        "structures": dataset.get("structures"),
        "materials": dataset.get("materials"),
        "created": dataset.get("created"),
        "open": dataset.get("open"),
        "static": dataset.get("static"),
        "license": dataset.get("license"),
        "method": dataset.get("method"),
        "links": dataset.get("links", [])[:4],
        "api_links": dataset.get("api_links", [])[:3],
    }


def build_llm_prompt(dataset_rows, task_contexts):
    payload = {
        "datasets_page": DATASETS_URL,
        "datasets": [compact_dataset_for_llm(row) for row in dataset_rows],
        "tasks": task_contexts,
        "ontology": {
            "dataset_classes": ["data:Dataset", "data:DatasetSplit", "data:LabeledDataset"],
            "evaluation_classes": ["eval:Task", "eval:ClassificationTask", "eval:RegressionTask"],
            "dataset_properties": [
                "data:hasNumberOfSamples",
                "data:wasDerivedFromDataset",
                "dcterms:source",
                "dcterms:description",
                "dcterms:license",
                "dcterms:created",
                "dcterms:references",
                "dcterms:accessRights",
            ],
            "evaluation_properties": ["eval:hasTask", "eval:usesDataset"],
        },
    }

    return f"""
You are helping populate a Turtle/RDF knowledge graph from Matbench Discovery pages.

Use ONLY the evidence in the JSON payload below. Keep identifiers stable.
Return one valid JSON object following this schema:

{{
  "datasets": [
    {{
      "id": "dataset id from input",
      "label": "human label",
      "ontology_class": "Dataset",
      "description": "short ontology-ready description in English",
      "source_url": "source page URL",
      "task_relevance": ["task ids where this dataset is directly useful"],
      "evidence": "short source pointer"
    }}
  ],
  "tasks": [
    {{
      "id": "task id from seed_task_id",
      "label": "human label",
      "ontology_class": "ClassificationTask or RegressionTask",
      "description": "short ontology-ready task definition",
      "uses_datasets": ["dataset ids"],
      "metric_types": ["metric names explicitly mentioned or implied by page"],
      "source_url": "task page URL",
      "evidence": "short source pointer"
    }}
  ]
}}

Hard rules:
- Use only ontology_class values: Dataset, ClassificationTask, RegressionTask.
- Do not invent dataset ids outside the supplied dataset ids and seed_dataset_ids.
- Preserve the five task ids from seed_task_id.
- Prefer concise, factual descriptions.
- Return JSON only. No markdown.

Payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def call_openrouter_json(prompt, model, output_json_path, max_retries=3):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise ontology information extraction assistant. "
                "Return only valid JSON. No markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    RAW_LLM_DIR.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/matbench-onto",
                    "X-Title": "Matbench Ontology Extraction",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "stream": False,
                },
                timeout=150,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            (RAW_LLM_DIR / f"{output_json_path.stem}_attempt_{attempt + 1}.txt").write_text(
                content or "",
                encoding="utf-8",
            )
            parsed = parse_json_response(content)
            output_json_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return parsed
        except (json.JSONDecodeError, requests.RequestException, KeyError) as error:
            last_error = error
            wait = 20 * (attempt + 1)
            print(f"OpenRouter retry in {wait}s after: {error}")
            time.sleep(wait)

    raise last_error


def parse_json_response(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content or "", re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def enrich_with_llm(dataset_rows, task_contexts, llm_json_path, model, use_cache=False):
    if use_cache and llm_json_path.exists():
        return json.loads(llm_json_path.read_text(encoding="utf-8"))
    prompt = build_llm_prompt(dataset_rows, task_contexts)
    return call_openrouter_json(prompt, model, llm_json_path)


def add_named(graph, uri, rdf_class, label):
    graph.add((uri, RDF.type, OWL.NamedIndividual))
    graph.add((uri, RDF.type, rdf_class))
    graph.add((uri, RDFS.label, Literal(label, datatype=XSD.string)))


def add_dataset(graph, dataset):
    uri = DATAIND[dataset["id"]]
    add_named(graph, uri, DATA.Dataset, dataset["label"])
    graph.add((uri, DCTERMS.source, URIRef(dataset.get("page") or DATASETS_URL)))

    if dataset.get("title") and dataset["title"] != dataset["label"]:
        graph.add((uri, DCTERMS.description, Literal(dataset["title"], datatype=XSD.string)))
    if dataset.get("structures") is not None:
        graph.add((uri, DATA.hasNumberOfSamples, Literal(dataset["structures"], datatype=XSD.integer)))
    if dataset.get("materials") is not None:
        graph.add((uri, DCTERMS.extent, Literal(f"{dataset['materials']} materials", datatype=XSD.string)))
    if dataset.get("created"):
        graph.add((uri, DCTERMS.created, Literal(dataset["created"], datatype=XSD.date)))
    if dataset.get("license"):
        graph.add((uri, DCTERMS.license, Literal(dataset["license"], datatype=XSD.string)))
    if dataset.get("method"):
        graph.add((uri, DCTERMS.description, Literal(f"Method: {dataset['method']}", datatype=XSD.string)))
    if dataset.get("open") is not None:
        graph.add((uri, DCTERMS.accessRights, Literal("open" if dataset["open"] else "closed", datatype=XSD.string)))
    if dataset.get("static") is not None:
        graph.add((uri, DCTERMS.description, Literal(f"Static dataset: {dataset['static']}", datatype=XSD.string)))
    for link in dataset.get("api_links", []) + dataset.get("links", []):
        graph.add((uri, DCTERMS.references, URIRef(link)))

    return uri


def add_llm_dataset_description(graph, dataset_data):
    raw_id = dataset_data.get("id")
    if not raw_id:
        return
    dataset_id = slugify(raw_id)
    uri = DATAIND[dataset_id]
    if (uri, RDF.type, None) not in graph:
        add_named(graph, uri, DATA.Dataset, dataset_data.get("label") or dataset_id)

    description = dataset_data.get("description")
    if description:
        graph.add((uri, DCTERMS.description, Literal(description, datatype=XSD.string)))
    source_url = dataset_data.get("source_url")
    if source_url:
        graph.add((uri, DCTERMS.source, URIRef(source_url)))


def add_llm_task(graph, task_data):
    raw_id = task_data.get("id")
    if not raw_id:
        return None
    task_id = slugify(raw_id)

    class_name = task_data.get("ontology_class")
    rdf_class = EVAL.ClassificationTask if class_name == "ClassificationTask" else EVAL.RegressionTask
    task_uri = EVALIND[task_id]
    add_named(graph, task_uri, rdf_class, task_data.get("label") or task_id)

    description = task_data.get("description")
    if description:
        graph.add((task_uri, RDFS.comment, Literal(description, datatype=XSD.string)))
    source_url = task_data.get("source_url")
    if source_url:
        graph.add((task_uri, DCTERMS.source, URIRef(source_url)))

    for dataset_id in task_data.get("uses_datasets", []) or []:
        dataset_uri = DATAIND[slugify(dataset_id)]
        graph.add((task_uri, EVAL.usesDataset, dataset_uri))

    for metric_name in task_data.get("metric_types", []) or []:
        metric_id = f"{slugify(metric_name)}_metric_type"
        metric_uri = EVALIND[metric_id]
        if (metric_uri, RDF.type, None) not in graph:
            add_named(graph, metric_uri, EVAL.MetricType, str(metric_name))

    return task_uri


def generate(output_path, use_llm=True, llm_json_path=DEFAULT_LLM_JSON, use_cache=False):
    graph = Graph()
    graph.bind("data", DATA)
    graph.bind("dataind", DATAIND)
    graph.bind("dcterms", DCTERMS)
    graph.bind("eval", EVAL)
    graph.bind("evalind", EVALIND)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)

    graph.add((URIRef("https://k.loria.fr/ontologies/matbench-site-context-individuals"), RDF.type, OWL.Ontology))

    dataset_rows, task_contexts = collect_scraped_context()
    for dataset in dataset_rows:
        add_dataset(graph, dataset)

    for dataset_id, label in SUPPORT_DATASETS.items():
        uri = DATAIND[dataset_id]
        if (uri, RDF.type, None) not in graph:
            add_named(graph, uri, DATA.Dataset, label)
        graph.add((uri, DCTERMS.source, URIRef(DATASETS_URL)))

    suite_uri = EVALIND["matbench_discovery_benchmark_suite"]
    release_uri = EVALIND["matbench_discovery_benchmark_release"]
    add_named(graph, suite_uri, EVAL.BenchmarkSuite, "Matbench Discovery")
    add_named(graph, release_uri, EVAL.BenchmarkRelease, "Matbench Discovery benchmark release")
    graph.add((suite_uri, EVAL.hasBenchmarkRelease, release_uri))

    if use_llm:
        model = configured_openrouter_model()
        print(f"Using OpenRouter model: {model}")
        llm_data = enrich_with_llm(dataset_rows, task_contexts, llm_json_path, model, use_cache=use_cache)
        for dataset_data in llm_data.get("datasets", []):
            add_llm_dataset_description(graph, dataset_data)
        task_entries = llm_data.get("tasks", [])
        for task_data in task_entries:
            task_uri = add_llm_task(graph, task_data)
            if task_uri is not None:
                graph.add((release_uri, EVAL.hasTask, task_uri))
        task_count = len(task_entries)
    else:
        task_count = 0
        for task_key, task_data in task_contexts.items():
            spec = TASK_SPECS[task_key]
            task_uri = EVALIND[spec["id"]]
            add_named(graph, task_uri, spec["class"], task_data["title"])
            graph.add((task_uri, DCTERMS.source, URIRef(task_data["url"])))
            graph.add((release_uri, EVAL.hasTask, task_uri))
            if task_data["description"]:
                graph.add((task_uri, RDFS.comment, Literal(task_data["description"], datatype=XSD.string)))
            for dataset_id in spec["datasets"]:
                graph.add((task_uri, EVAL.usesDataset, DATAIND[dataset_id]))
            task_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_path, format="turtle")
    return len(dataset_rows), task_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Matbench Discovery dataset/task pages and generate ontology individuals."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-json", type=Path, default=DEFAULT_LLM_JSON)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_count, task_count = generate(
        args.output.resolve(),
        use_llm=not args.no_llm,
        llm_json_path=args.llm_json.resolve(),
        use_cache=args.use_cache,
    )
    print(f"Generated {args.output}")
    print(f"Datasets scraped: {dataset_count}")
    print(f"Tasks added: {task_count}")


if __name__ == "__main__":
    main()
