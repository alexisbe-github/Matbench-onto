import json
import math
import re
import yaml
from pypdf import PdfReader
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from openrouter import OpenRouter, errors


BASE_DIR = Path(__file__).resolve().parents[3]

load_dotenv(BASE_DIR / ".env")

ONTOLOGY_FILES = [
    BASE_DIR / "ontology/architecture.ttl",
    BASE_DIR / "ontology/trainingonto.ttl",
    BASE_DIR / "ontology/datasetonto.ttl",
    BASE_DIR / "ontology/evaluationonto.ttl",
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "FREE_OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)

# Keep one free-model request per paper, but retrieve a small local context for
# every question instead of sending the first 30 PDF pages wholesale.
CONTEXT_CHUNK_CHARS = int(os.getenv("CONTEXT_CHUNK_CHARS", "900"))
CONTEXT_PDF_TOP_K = int(os.getenv("CONTEXT_PDF_TOP_K", "2"))
CONTEXT_YAML_TOP_K = int(os.getenv("CONTEXT_YAML_TOP_K", "4"))
CONTEXT_ONTOLOGY_TOP_K = int(os.getenv("CONTEXT_ONTOLOGY_TOP_K", "2"))
HYPERPARAMETER_PDF_TOP_K = int(os.getenv("HYPERPARAMETER_PDF_TOP_K", "2"))
REQUEST_DELAY_SECONDS = float(os.getenv("OPENROUTER_REQUEST_DELAY_SECONDS", "0"))

YAML_FILE = Path(os.environ["YAML_FILE"])

PROMPT_FILE = Path(
    os.getenv(
        "PROMPT_FILE",
        Path(__file__).resolve().parent / "prompts.json",
    )
)

PDF_FILE = Path(os.environ["PDF_FILE"])
PDF_URL = os.getenv("PDF_URL")
MODEL_PAGE_FILE = Path(os.environ["MODEL_PAGE_FILE"]) if os.getenv("MODEL_PAGE_FILE") else None
MODEL_PAGE_URL = os.getenv("MODEL_PAGE_URL")

OUTPUT_JSON_FILE = Path(
    os.getenv(
        "OUTPUT_JSON_FILE",
        BASE_DIR / "outputs" / "model_extraction.json"
    )
)
OUTPUT_TTL_FILE = BASE_DIR / "outputs" / "model_llm.ttl"

TARGET_ENTITY_TYPES = [
    "ModelFamily",
    "ModelVariant",
    "MachineLearningArchitecture",
    "MachineLearningArchitectureConfiguration",
    "MachineLearningTrainingRun",
    "TrainingDataset",
    "Checkpoint",
    "Optimizer",
    "LossFunction",
    "MachineLearningHyperparameter",
    "MachineLearningHyperparameterValue"
]

EXPECTED_OUTPUT_SCHEMA = {
    "model_name": None,
    "model_family": None,
    "model_variant": None,
    "architecture": None,
    "parameter_number": None
}

CLIENT = None

STOP_WORDS = {
    "a", "an", "and", "are", "as", "be", "by", "does", "for", "from",
    "how", "if", "in", "is", "it", "its", "of", "only", "or", "return",
    "such", "that", "the", "this", "to", "used", "value", "what", "which",
    "with", "item", "null", "name", "exact", "reported",
}

QUERY_EXPANSIONS = {
    "architecture": "network model layer backbone head encoder decoder message passing graph transformer",
    "dataset": "data training validation test samples structures materials configurations corpus split",
    "evaluation": "evaluate benchmark metrics score performance accuracy mae rmse discovery test task",
    "hyperparameter": "learning rate batch epoch cutoff optimizer hidden dimension layers training settings",
    "materialization": "checkpoint pretrained weights model release output",
    "training": "train pretraining fine tuning loss optimizer objective sampling initialization distillation",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_pdf_text(path):
    reader = PdfReader(path)
    pages = []

    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": index + 1,
                "text": text
            })

    return pages


def load_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_ontology_context(paths):
    ontology_parts = []

    for path in paths:
        if path.exists():
            ontology_parts.append({
                "file": path.name,
                "content": load_text(path)
            })

    return ontology_parts


def tokenize(text):
    # Make ontology identifiers searchable: MachineLearningHyperparameter ->
    # Machine Learning Hyperparameter, and snake_case -> separate tokens.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    text = text.replace("_", " ")
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def expand_query(text):
    expanded = str(text)
    lowered = expanded.lower()
    for key, words in QUERY_EXPANSIONS.items():
        if key in lowered or f"{key}s" in lowered:
            expanded += " " + words
    return expanded


def chunk_text(text, max_chars=CONTEXT_CHUNK_CHARS):
    """Split extracted text into readable chunks without cutting lines."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    chunks = []
    current = []
    current_size = 0
    for line in lines:
        if current and current_size + len(line) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-1:]
            current_size = sum(len(part) + 1 for part in current)
        current.append(line)
        current_size += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_pdf_chunks(pdf_pages):
    chunks = []
    for page in pdf_pages:
        for index, content in enumerate(chunk_text(page["text"]), start=1):
            chunks.append({
                "id": f"P{page['page']}-C{index}", "kind": "pdf",
                "page": page["page"], "text": content,
            })
    return chunks


def build_ontology_chunks(ontology_context):
    chunks = []
    for ontology in ontology_context:
        for index, content in enumerate(chunk_text(ontology["content"]), start=1):
            stem = re.sub(r"\W+", "_", ontology["file"]).strip("_")
            chunks.append({
                "id": f"O-{stem}-{index}", "kind": "ontology",
                "file": ontology["file"], "text": content,
            })
    return chunks


def build_ontology_class_catalog(ontology_context):
    """Build a compact authoritative list used by every ontology mapping loop."""
    classes = {}
    declaration_pattern = re.compile(
        r"(?::|#)([A-Za-z][A-Za-z0-9_-]*)>?\s+rdf:type\s+owl:Class\b"
    )
    for ontology in ontology_context:
        for class_name in declaration_pattern.findall(ontology["content"]):
            classes.setdefault(class_name, ontology["file"])
    lines = [
        f"{class_name} | {file_name} | declared rdf:type owl:Class"
        for class_name, file_name in sorted(classes.items())
    ]
    return {
        "id": "O-CLASS-CATALOG",
        "kind": "ontology",
        "file": "ontology class catalog",
        "text": "\n".join(lines),
        "classes": classes,
    }


def flatten_yaml(data, path=""):
    facts = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)
            facts.extend(flatten_yaml(value, child_path))
    elif isinstance(data, list):
        if len(data) <= 12 and all(not isinstance(item, (dict, list)) for item in data):
            facts.append((path, data))
        else:
            for index, value in enumerate(data):
                facts.extend(flatten_yaml(value, f"{path}[{index}]"))
    else:
        facts.append((path, data))
    return facts


def build_yaml_chunks(yaml_data):
    return [
        {
            "id": f"Y{index}", "kind": "yaml", "path": path,
            "text": f"{path} = {json.dumps(value, ensure_ascii=False)}",
        }
        for index, (path, value) in enumerate(flatten_yaml(yaml_data), start=1)
    ]


def parse_model_page_hyperparameters(model_page_context):
    """Parse the already-rendered Hyperparams section saved by the scraper."""
    if not isinstance(model_page_context, dict):
        return {}
    text = " ".join(
        section.get("text", "")
        for section in model_page_context.get("sections", [])
        if isinstance(section, dict)
    )
    match = re.search(r"\bHyperparams?(?:eters)?\b", text, re.IGNORECASE)
    if not match:
        return {}
    section = text[match.end():]
    section = re.split(r"\bDependencies\b", section, maxsplit=1, flags=re.IGNORECASE)[0]
    decoder = json.JSONDecoder()
    field_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*")
    parsed = {}
    position = 0

    while True:
        field = field_pattern.search(section, position)
        if not field:
            break
        key = field.group(1)
        value_start = field.end()
        while value_start < len(section) and section[value_start].isspace():
            value_start += 1
        if value_start >= len(section):
            break

        if section[value_start] in "{[":
            try:
                value, consumed = decoder.raw_decode(section[value_start:])
                parsed[key] = value
                position = value_start + consumed
                continue
            except json.JSONDecodeError:
                pass

        next_field = field_pattern.search(section, value_start)
        value_end = next_field.start() if next_field else len(section)
        raw_value = section[value_start:value_end].strip()
        if raw_value:
            try:
                parsed[key] = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                parsed[key] = raw_value.strip('"')
        position = value_end

    return parsed


def flatten_hyperparameters(data, path="hyperparams"):
    facts = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)
            facts.extend(flatten_hyperparameters(value, child_path))
    else:
        facts.append({
            "path": path,
            "name": path.rsplit(".", 1)[-1],
            "category": path.split(".", 2)[1] if path.count(".") >= 2 else None,
            "value": data,
        })
    return facts


def build_known_hyperparameters(model_page_context, yaml_data):
    """Use scraped page hyperparameters first, with YAML as a lossless fallback."""
    page_hyperparameters = parse_model_page_hyperparameters(model_page_context)
    source_kind = "model_page"
    source_url = (
        (model_page_context or {}).get("url")
        or MODEL_PAGE_URL
    )
    if not page_hyperparameters:
        page_hyperparameters = yaml_data.get("hyperparams") or {}
        source_kind = "yaml"
        source_url = None

    facts = flatten_hyperparameters(page_hyperparameters)
    chunks = []
    for index, fact in enumerate(facts, start=1):
        chunk_id = f"M-HYP-{index}" if source_kind == "model_page" else f"HY-HYP-{index}"
        label = f"{fact['path']} = {json.dumps(fact['value'], ensure_ascii=False)}"
        chunks.append({
            "id": chunk_id,
            "kind": source_kind,
            "path": fact["path"],
            "name": fact["name"],
            "category": fact["category"],
            "value": fact["value"],
            "source": source_url,
            "text": label,
        })
    return chunks


def flatten_questions(node, path=()):
    questions = []
    if isinstance(node, dict):
        if "question" in node and "variable" in node:
            questions.append({
                "id": path[-1] if path else node["variable"],
                "path": ".".join(path), "question": node["question"],
                "variable": node["variable"],
                "source": node.get("source", "document"),
                "yaml_path": node.get("yaml_path"),
            })
        for key, value in node.items():
            if key not in {"question", "variable", "source", "yaml_path"}:
                questions.extend(flatten_questions(value, path + (str(key),)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            questions.extend(flatten_questions(value, path + (str(index),)))
    return questions


def rank_chunks(query, chunks, top_k, expand=True):
    """Small BM25-like ranker; it avoids an embedding/API round trip."""
    if not chunks or top_k <= 0:
        return []
    query_tokens = tokenize(expand_query(query) if expand else query)
    document_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    frequencies = {}
    for tokens in document_tokens:
        for token in set(tokens):
            frequencies[token] = frequencies.get(token, 0) + 1
    scored = []
    document_count = len(chunks)
    for chunk, tokens in zip(chunks, document_tokens):
        token_set = set(tokens)
        score = sum(
            math.log(1 + (document_count + 1) / (frequencies[token] + 1))
            for token in query_tokens if token in token_set
        )
        score /= 1 + max(0, len(tokens) - 120) / 500
        scored.append((score, chunk["id"], chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k] if item[0] > 0]


def build_question_contexts(
    questions, yaml_data, pdf_pages, ontology_context, model_page_context=None
):
    pdf_chunks = build_pdf_chunks(pdf_pages)
    yaml_chunks = build_yaml_chunks(yaml_data)
    ontology_chunks = build_ontology_chunks(ontology_context)
    ontology_catalog = build_ontology_class_catalog(ontology_context)
    known_hyperparameter_chunks = build_known_hyperparameters(
        model_page_context, yaml_data
    )
    yaml_by_path = {chunk.get("path"): chunk for chunk in yaml_chunks}
    used_chunks = {}
    packets = []
    for question in questions:
        selected = []
        if question["source"] == "ontology_mapping":
            # Parent loop paths (for example "architectures") must not outweigh
            # the actual item kind (for example "hyperparameter").
            selected.append(ontology_catalog)
            selected.extend(
                rank_chunks(
                    question["question"], ontology_chunks,
                    CONTEXT_ONTOLOGY_TOP_K, expand=False,
                )
            )
            if question["variable"] in {
                "backbone_ontology_class", "head_ontology_class"
            }:
                selected.extend(
                    rank_chunks(
                        question["question"], pdf_chunks,
                        CONTEXT_PDF_TOP_K,
                    )
                )
        else:
            query = " ".join(
                (question["path"], question["variable"], question["question"])
            )
            direct_yaml = yaml_by_path.get(question.get("yaml_path"))
            if direct_yaml:
                selected.append(direct_yaml)
            selected.extend(rank_chunks(query, yaml_chunks, CONTEXT_YAML_TOP_K))
            selected.extend(rank_chunks(query, pdf_chunks, CONTEXT_PDF_TOP_K))
        context_ids = []
        for chunk in selected:
            if chunk["id"] not in context_ids:
                context_ids.append(chunk["id"])
                used_chunks[chunk["id"]] = chunk
        packets.append({**question, "context_ids": context_ids})

    # Hyperparameters are known before the LLM call. Search the paper once per
    # exact name/value pair instead of using the generic {item} loop question.
    for hyperparameter in known_hyperparameter_chunks:
        value_text = json.dumps(hyperparameter["value"], ensure_ascii=False)
        query = (
            f"{hyperparameter['category'] or ''} {hyperparameter['name']} "
            f"{value_text} hyperparameter training"
        )
        selected = [hyperparameter]
        selected.extend(
            rank_chunks(
                query, pdf_chunks, HYPERPARAMETER_PDF_TOP_K, expand=False
            )
        )
        context_ids = []
        for chunk in selected:
            if chunk["id"] not in context_ids:
                context_ids.append(chunk["id"])
                used_chunks[chunk["id"]] = chunk
        packets.append({
            "id": f"known_hyperparameter::{hyperparameter['path']}",
            "path": f"known_hyperparameters.{hyperparameter['path']}",
            "question": (
                f"Find paper evidence for {hyperparameter['path']} = {value_text}."
            ),
            "variable": "known_hyperparameter",
            "source": "known_hyperparameter",
            "yaml_path": None,
            "known_hyperparameter": {
                "path": hyperparameter["path"],
                "name": hyperparameter["name"],
                "category": hyperparameter["category"],
                "value": hyperparameter["value"],
                "authoritative_context_id": hyperparameter["id"],
            },
            "context_ids": context_ids,
        })
    return packets, list(used_chunks.values())


def build_yaml_summary(yaml_data):
    return {
        "model_name": yaml_data.get("model_name"),
        "model_key": yaml_data.get("model_key"),
        "model_type": yaml_data.get("model_type"),
        "model_params": yaml_data.get("model_params"),
        "notes_description": yaml_data.get("notes", {}).get("Description")
    }


def build_pdf_context(pdf_pages):
    return "\n\n".join(page["text"] for page in pdf_pages[:30])


def get_questions(prompts, yaml_data):
    return prompts["questionnaire"]


def clean_llm_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    decoder = json.JSONDecoder()
    start = text.find("{")

    if start == -1:
        return text

    obj, end = decoder.raw_decode(text[start:])
    return json.dumps(obj, ensure_ascii=False)


def query_openrouter(prompt, json_format=True, max_retries=6):
    global CLIENT
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY missing in .env file")
    if CLIENT is None:
        CLIENT = OpenRouter(api_key=OPENROUTER_API_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise information extraction assistant. "
                "Return only valid JSON when JSON is requested. "
                "Do not wrap JSON in markdown."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if json_format:
        messages[1]["content"] += "\n\nReturn ONLY valid JSON. No markdown."

    last_error = None

    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.send(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0,
                stream=False
            )

            if REQUEST_DELAY_SECONDS:
                time.sleep(REQUEST_DELAY_SECONDS)
            return response.choices[0].message.content

        except errors.TooManyRequestsResponseError as error:
            last_error = error
            wait_time = 30 * (attempt + 1)
            print(f"Rate limit OpenRouter. Retry in {wait_time}s...")
            time.sleep(wait_time)

    raise last_error


def format_context_chunk(chunk):
    if chunk["kind"] == "pdf":
        label = f"PDF page {chunk['page']}"
    elif chunk["kind"] == "yaml":
        label = f"YAML path {chunk['path']}"
    elif chunk["kind"] == "model_page":
        label = f"official model page field {chunk['path']}"
    elif chunk["kind"] == "yaml_hyperparameter":
        label = f"YAML hyperparameter {chunk['path']}"
    else:
        label = f"ontology file {chunk['file']}"
    return f"[{chunk['id']}] {label}\n{chunk['text']}"


def build_combined_extraction_prompt(prompts, questions, context_chunks):
    contexts = "\n\n".join(format_context_chunk(chunk) for chunk in context_chunks)
    retrieval_map = [
        {
            "question_id": question["id"],
            "question_path": question["path"],
            "context_ids": question["context_ids"],
            **(
                {"known_hyperparameter": question["known_hyperparameter"]}
                if question.get("known_hyperparameter") else {}
            ),
        }
        for question in questions
    ]

    return f"""
Follow the nested questionnaire and return the same nested extraction structure.

Questionnaire:
{json.dumps(prompts["questionnaire"], separators=(",", ":"), ensure_ascii=False)}

Question-specific retrieval map (use only the listed context_ids for that question):
{json.dumps(retrieval_map, separators=(",", ":"), ensure_ascii=False)}

Context bank:
{contexts}

Return one valid nested JSON object.

Expected top-level keys:
- model
- architectures
- training
- datasets
- evaluation

Rules:
- model must be an object.
- Put training runs under training.training_runs and evaluation runs under evaluation.evaluation_runs.
- Do not return flat keys.
- Architectures must contain one object per architecture.
- if multiple architectures are mentioned, NEVER merge them.
- preserve all architectures found in the paper.
- each architecture object must contain its own fields.
- Use YAML evidence first when it directly answers a question.
- The known_hyperparameter entries were parsed before this call from the official model
  page (or YAML fallback). They are authoritative: include every one with its exact value.
- For a known hyperparameter value, cite its authoritative M-HYP/HY-HYP context as evidence.
  A retrieved PDF passage is optional supporting paper_evidence, never a replacement value.
- Use PDF passages only when YAML is insufficient.
- Use ontology passages only for ontology class mapping.
- Use null when unknown.
- Every extracted field must be {{"value": ..., "evidence": ...}} (plus "source" when useful).
- PDF evidence MUST be a short verbatim quote copied from one listed passage, in this exact form:
  PDF p. <page> [<passage-id>]: "<short exact quote>"
- YAML evidence MUST contain the exact YAML path and value.
- Model-page evidence MUST use: Model page [M-HYP-<n>]: <exact path> = <exact value>.
- Ontology evidence MUST contain the ontology filename, class/property and passage id.
- For ontology_mapping, use only a class explicitly listed in [O-CLASS-CATALOG].
- Every non-null backbone_architecture must have backbone_ontology_class, and every
  non-null head_architecture must have head_ontology_class. These class fields describe
  the named component individual; they are not replacements for its name.
- Never specialize a generic class from memory. If LearningRate is absent from the catalog,
  use the declared MachineLearningHyperparameter class rather than inventing LearningRate.
- An evidence is proof, not a paraphrase. Never invent a page, passage id, or quote.
- For null/empty answers, use evidence "Not found in the retrieved contexts".
"""


def best_sentence(text, hint):
    sentences = re.split(r"(?<=[.!?])\s+|\s*[;]\s*", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return text[:300].strip()
    hint_tokens = set(tokenize(hint))
    return max(
        sentences,
        key=lambda sentence: len(hint_tokens.intersection(tokenize(sentence))),
    )[:350]


def refine_evidence_citations(data, context_chunks, pdf_url=None):
    """Turn cited PDF passages into short, exact and immediately findable proof."""
    chunk_by_id = {chunk["id"]: chunk for chunk in context_chunks}
    ontology_catalog = chunk_by_id.get("O-CLASS-CATALOG", {})
    ontology_classes = ontology_catalog.get("classes", {})
    citation_pattern = re.compile(r"\b(?:P\d+-C\d+|Y\d+|O-[A-Za-z0-9_]+-\d+)\b")
    model_page_pattern = re.compile(r"\b(?:M-HYP|HY-HYP)-\d+\b")

    def visit(node, path=()):
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, path + (str(index),))
            return
        if not isinstance(node, dict):
            return
        if "value" in node and "evidence" in node:
            evidence = str(node.get("evidence") or "")
            cited_hyperparameter = next(
                (
                    chunk_by_id[citation_id]
                    for citation_id in model_page_pattern.findall(evidence)
                    if citation_id in chunk_by_id
                ),
                None,
            )
            if cited_hyperparameter:
                if cited_hyperparameter["id"].startswith("M-HYP"):
                    node["evidence"] = (
                        f"Model page [{cited_hyperparameter['id']}]: "
                        f"{cited_hyperparameter['text']}"
                    )
                    if cited_hyperparameter.get("source"):
                        node["source"] = cited_hyperparameter["source"]
                else:
                    node["evidence"] = (
                        f"YAML [{cited_hyperparameter['id']}]: "
                        f"{cited_hyperparameter['text']}"
                    )
                evidence = node["evidence"]
            ontology_class_fields = {
                "ontology_class", "backbone_ontology_class", "head_ontology_class"
            }
            if path and path[-1] in ontology_class_fields and ontology_classes:
                class_name = str(node.get("value") or "").split(":")[-1]
                if class_name not in ontology_classes:
                    path_text = ".".join(path).lower()
                    fallbacks = (
                        ("hyperparameter", "MachineLearningHyperparameter"),
                        ("dataset", "Dataset"),
                        ("backbone", "MachineLearningArchitecture"),
                        ("head", "MachineLearningArchitecture"),
                        ("architecture", "MachineLearningArchitecture"),
                    )
                    class_name = next(
                        (
                            fallback for marker, fallback in fallbacks
                            if marker in path_text and fallback in ontology_classes
                        ),
                        None,
                    )
                    node["value"] = class_name
                if class_name in ontology_classes:
                    ontology_file = ontology_classes[class_name]
                    node["evidence"] = (
                        f"Ontology {ontology_file} [O-CLASS-CATALOG]: "
                        f"{class_name} is declared rdf:type owl:Class."
                    )
                    evidence = node["evidence"]

            cited_pdf = next(
                (
                    chunk_by_id[citation_id]
                    for citation_id in citation_pattern.findall(evidence)
                    if citation_id in chunk_by_id
                    and chunk_by_id[citation_id]["kind"] == "pdf"
                ),
                None,
            )
            if cited_pdf:
                quote = best_sentence(
                    cited_pdf["text"], f"{node.get('value')} {evidence}"
                )
                quoted = f'"{quote}"' if '"' not in quote else f"'{quote}'"
                node["evidence"] = (
                    f"PDF p. {cited_pdf['page']} [{cited_pdf['id']}]: {quoted}"
                )
                if pdf_url:
                    node["source"] = pdf_url
        for key, value in node.items():
            visit(value, path + (str(key),))

    visit(data)
    return data


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_ttl(ttl_content, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(ttl_content)


def main():
    yaml_data = load_yaml(YAML_FILE)
    ontology_context = load_ontology_context(ONTOLOGY_FILES)
    prompts = load_json(PROMPT_FILE)
    pdf_pages = load_pdf_text(PDF_FILE)
    model_page_context = (
        load_json(MODEL_PAGE_FILE)
        if MODEL_PAGE_FILE and MODEL_PAGE_FILE.exists()
        else None
    )
    questions = flatten_questions(prompts["questionnaire"])
    question_packets, context_chunks = build_question_contexts(
        questions, yaml_data, pdf_pages, ontology_context, model_page_context
    )
    hyperparameter_packets = [
        packet for packet in question_packets if packet.get("known_hyperparameter")
    ]

    full_pdf_chars = sum(len(page["text"]) for page in pdf_pages)
    retrieved_chars = sum(len(chunk["text"]) for chunk in context_chunks)
    print(
        f"Context retrieval: {len(questions)} questionnaire questions, "
        f"{len(hyperparameter_packets)} known hyperparameters, "
        f"{len(context_chunks)} unique chunks, {retrieved_chars:,} chars "
        f"(paper: {full_pdf_chars:,} chars)."
    )

    retrieval_dir = OUTPUT_JSON_FILE.parent.parent / "retrieval"
    model_stem = OUTPUT_JSON_FILE.stem.removesuffix("_model_extraction")
    retrieval_file = retrieval_dir / f"{model_stem}_retrieval.json"
    save_json({
        "question_contexts": [
            {
                "question_id": packet["id"],
                "question": packet["question"],
                "known_hyperparameter": packet.get("known_hyperparameter"),
                "context_ids": packet["context_ids"],
            }
            for packet in question_packets
        ],
        "context_bank": context_chunks,
    }, retrieval_file)
    print(f"Retrieval map: {retrieval_file}")

    extraction_prompt = build_combined_extraction_prompt(
        prompts, question_packets, context_chunks
    )
    response = query_openrouter(extraction_prompt)
    results = json.loads(clean_llm_json(response))
    refine_evidence_citations(results, context_chunks, PDF_URL)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    results["_sources"] = {
        "pdf_file": str(PDF_FILE),
        "pdf_url": PDF_URL,
        "yaml_file": str(YAML_FILE),
        "model_page_file": str(MODEL_PAGE_FILE) if MODEL_PAGE_FILE else None,
        "model_page_url": (
            (model_page_context or {}).get("url") or MODEL_PAGE_URL
        ),
        "retrieval": {
            "questions": len(questions),
            "known_hyperparameters": len(hyperparameter_packets),
            "unique_context_chunks": len(context_chunks),
            "context_characters": retrieved_chars,
            "paper_characters": full_pdf_chars,
            "file": str(retrieval_file),
        },
    }
    save_json(results, OUTPUT_JSON_FILE)


if __name__ == "__main__":
    main()
