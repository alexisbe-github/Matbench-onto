import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent

YAML_FILE = BASE_DIR / "mattersim-v1-5M.yml"
PROMPT_FILE = BASE_DIR / "prompts.json"
ONTOLOGY_CONTEXT_FILE = BASE_DIR / "ontology_context.json"
PDF_FILE = BASE_DIR / "2405.04967v2.pdf"

OUTPUT_FILE = BASE_DIR / "outputs" / "model_extraction.json"
DEBUG_DIR = BASE_DIR / "debug"

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

MAX_CANDIDATES = 50
USE_DIRECT_MATCH = True
USE_LLM = True


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_path(data, path):
    if not path:
        return None

    current = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)

    return current


def flatten_yaml(data, prefix=""):
    result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_yaml(value, next_prefix))
    else:
        result[prefix] = data

    return result


def yaml_to_text(yaml_data):
    flat = flatten_yaml(yaml_data)
    lines = []

    for key, value in flat.items():
        if value is not None:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def read_pdf_text(path):
    if not path.exists():
        return ""

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []

        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"PAGE {index + 1}\n{text}")

        return "\n\n".join(pages)

    except Exception:
        return ""


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def split_identifier(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"[_\-/#:.,;()\[\]{}]", " ", text)
    return text


def tokens(text):
    text = split_identifier(text)
    return set(
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", text)
    )


def score_text(candidate_text, query_text):
    candidate_tokens = tokens(candidate_text)
    query_tokens = tokens(query_text)

    if not candidate_tokens or not query_tokens:
        return 0

    score = len(candidate_tokens.intersection(query_tokens))

    lower_candidate = clean_text(candidate_text).lower()
    lower_query = clean_text(query_text).lower()

    if lower_candidate and lower_candidate in lower_query:
        score += 10

    for token in candidate_tokens:
        if len(token) >= 3 and token in lower_query:
            score += 2

    return score


def chunk_text(text, size=2500, overlap=300):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def retrieve_pdf_context(pdf_text, query, top_k=4):
    chunks = chunk_text(pdf_text)

    if not chunks:
        return ""

    ranked = sorted(
        chunks,
        key=lambda chunk: score_text(chunk, query),
        reverse=True,
    )

    return "\n\n".join(ranked[:top_k])


def individual_text(individual):
    fields = [
        individual.get("name", ""),
        individual.get("label", ""),
        individual.get("iri", ""),
    ]

    for rdf_type in individual.get("types", []):
        if isinstance(rdf_type, dict):
            fields.append(rdf_type.get("name", ""))
            fields.append(rdf_type.get("iri", ""))

    return " ".join(str(field) for field in fields if field)


def expected_terms(question_id, question_data):
    text = " ".join([
        question_id,
        question_data.get("question", ""),
        question_data.get("variable", ""),
        question_data.get("yaml_path", ""),
    ]).lower()

    terms = []

    if "family" in text:
        terms.extend(["ModelFamily"])

    if "variant" in text:
        terms.extend(["ModelVariant"])

    if "architecture" in text:
        terms.extend([
            "Architecture",
            "MachineLearningArchitecture",
            "NeuralNetworkArchitecture",
            "GraphNeuralNetworkArchitecture",
            "MessagePassingNeuralNetworkArchitecture",
            "TransformerArchitecture",
            "GraphTransformerArchitecture",
            "EquivariantNeuralNetworkArchitecture",
            "EquivariantGraphNeuralNetworkArchitecture",
            "EquivariantMessagePassingNeuralNetworkArchitecture",
        ])

    if "dataset" in text or "training_set" in text:
        terms.extend([
            "Dataset",
            "TrainingDataset",
            "LabeledDataset",
        ])

    if "optimizer" in text:
        terms.extend(["Optimizer"])

    if "loss" in text:
        terms.extend(["LossFunction", "ObjectiveFunction"])

    if "training" in text:
        terms.extend([
            "MachineLearningTrainingRun",
            "PretrainingRun",
            "MachineLearningFinetuningRun",
            "MachineLearningActiveLearningRun",
        ])

    if "checkpoint" in text:
        terms.extend(["Checkpoint"])

    if "hyperparameter" in text:
        terms.extend([
            "MachineLearningHyperparameter",
            "MachineLearningHyperparameterValue",
        ])

    if "algorithm" in text or "method" in text:
        terms.extend(["AlgorithmMethod"])

    return terms


def matches_expected_terms(individual, terms):
    if not terms:
        return True

    text = individual_text(individual).lower()

    for term in terms:
        if term.lower() in text:
            return True

    return False


def shortlist_individuals(individuals, question_id, question_data, raw_value, yaml_context, pdf_context):
    terms = expected_terms(question_id, question_data)

    query = "\n".join([
        question_id,
        question_data.get("question", ""),
        question_data.get("variable", ""),
        question_data.get("yaml_path", ""),
        str(raw_value),
        yaml_context[:3000],
        pdf_context[:3000],
    ])

    filtered = [
        individual for individual in individuals
        if matches_expected_terms(individual, terms)
    ]

    if len(filtered) < 5:
        filtered = individuals

    ranked = sorted(
        filtered,
        key=lambda individual: score_text(individual_text(individual), query),
        reverse=True,
    )

    return ranked[:MAX_CANDIDATES]


def direct_match(raw_value, candidates):
    if raw_value is None:
        return None

    raw = clean_text(raw_value).lower()

    if not raw:
        return None

    best = None
    best_score = 0

    for candidate in candidates:
        name = clean_text(candidate.get("name", "")).lower()
        label = clean_text(candidate.get("label", "")).lower()
        iri = clean_text(candidate.get("iri", "")).lower()

        names = [name, label]

        if "#" in iri:
            names.append(iri.split("#")[-1])
        else:
            names.append(iri.rstrip("/").split("/")[-1])

        score = 0

        for value in names:
            value = clean_text(value).lower()

            if not value:
                continue

            if value in raw:
                score += 20

            value_tokens = tokens(value)
            raw_tokens = tokens(raw)

            common = value_tokens.intersection(raw_tokens)
            score += len(common) * 4

            if value_tokens and value_tokens.issubset(raw_tokens):
                score += 15

        if score > best_score:
            best = candidate
            best_score = score

    if best and best_score >= 8:
        return {
            "selected_iri": best.get("iri"),
            "selected_name": best.get("name"),
            "selected_individual": best,
            "confidence": 0.95,
            "source": "yaml",
            "evidence": str(raw_value),
            "reason": "Direct match between the YAML value and an existing ontology individual.",
        }

    return None


def build_prompt(question_id, question_data, raw_value, yaml_context, pdf_context, candidates):
    payload = {
        "task": "Select the existing RDF individual that best answers the question.",
        "rules": [
            "Select only from candidate_individuals.",
            "Do not invent IRIs.",
            "Do not create new individuals.",
            "If no existing individual clearly matches, selected_iri must be null.",
            "Use the YAML first.",
            "Use the PDF only if the YAML is missing, ambiguous, or insufficient.",
            "Return only valid JSON.",
            "The returned JSON must contain exactly these keys: selected_iri, selected_name, confidence, source, evidence, reason."
        ],
        "question": {
            "id": question_id,
            "text": question_data.get("question"),
            "variable": question_data.get("variable"),
            "yaml_path": question_data.get("yaml_path"),
            "raw_yaml_value": raw_value,
        },
        "yaml_context": yaml_context[:7000],
        "pdf_context": pdf_context[:9000],
        "candidate_individuals": candidates,
        "output_format": {
            "selected_iri": "exact IRI of a candidate individual or null",
            "selected_name": "candidate name or null",
            "confidence": "number between 0 and 1",
            "source": "yaml | pdf | yaml+pdf | none",
            "evidence": "short evidence from YAML or PDF",
            "reason": "short reason for the choice or non-choice"
        }
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse_json(text):
    text = str(text).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError("The model did not return a JSON object.")

    return parsed


def call_ollama(prompt, debug_name):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "You are an RDF/OWL entity linking system.\n"
            "Return only valid JSON with exactly these keys:\n"
            "selected_iri, selected_name, confidence, source, evidence, reason.\n\n"
            + prompt
        ),
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc

    raw = result.get("response", "")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / f"{debug_name}_raw.txt").write_text(raw, encoding="utf-8")
    (DEBUG_DIR / f"{debug_name}_prompt.txt").write_text(prompt, encoding="utf-8")

    parsed = parse_json(raw)

    return {
        "selected_iri": parsed.get("selected_iri"),
        "selected_name": parsed.get("selected_name"),
        "confidence": parsed.get("confidence"),
        "source": parsed.get("source"),
        "evidence": parsed.get("evidence"),
        "reason": parsed.get("reason"),
        "_raw_response": raw,
    }


def validate_answer(answer, candidates):
    allowed = {candidate["iri"]: candidate for candidate in candidates}
    selected_iri = answer.get("selected_iri")

    if selected_iri in ["", "null", "None", "none"]:
        selected_iri = None

    if selected_iri is None:
        answer["selected_iri"] = None
        answer["selected_name"] = None
        answer["selected_individual"] = None
        return answer

    if selected_iri not in allowed:
        return {
            "selected_iri": None,
            "selected_name": None,
            "selected_individual": None,
            "confidence": 0,
            "source": "none",
            "evidence": "",
            "reason": f"Returned IRI is not present in candidates: {selected_iri}",
        }

    selected = allowed[selected_iri]
    answer["selected_iri"] = selected_iri
    answer["selected_name"] = selected.get("name")
    answer["selected_individual"] = selected

    return answer


def empty_answer(reason):
    return {
        "selected_iri": None,
        "selected_name": None,
        "selected_individual": None,
        "confidence": 0,
        "source": "none",
        "evidence": "",
        "reason": reason,
    }


def run_extraction():
    yaml_data = read_yaml(YAML_FILE)
    template = read_json(PROMPT_FILE)
    ontology_context = read_json(ONTOLOGY_CONTEXT_FILE)

    individuals = ontology_context.get("individuals", [])
    yaml_context = yaml_to_text(yaml_data)
    pdf_text = read_pdf_text(PDF_FILE)

    output = {
        "template_id": template["template_id"],
        "model_file": str(YAML_FILE),
        "pdf_file": str(PDF_FILE),
        "ontology_context_file": str(ONTOLOGY_CONTEXT_FILE),
        "ollama_model": OLLAMA_MODEL,
        "num_individuals": len(individuals),
        "answers": {},
    }

    for section_name, questions in template["questionnaire"].items():
        output["answers"][section_name] = {}

        for question_id, question_data in questions.items():
            if not question_data:
                continue

            raw_value = get_path(yaml_data, question_data.get("yaml_path"))

            pdf_query = " ".join([
                question_id,
                question_data.get("question", ""),
                question_data.get("variable", ""),
                str(raw_value),
            ])

            pdf_context = retrieve_pdf_context(pdf_text, pdf_query)

            candidates = shortlist_individuals(
                individuals=individuals,
                question_id=question_id,
                question_data=question_data,
                raw_value=raw_value,
                yaml_context=yaml_context,
                pdf_context=pdf_context,
            )

            debug_name = f"{section_name}_{question_id}"

            try:
                direct = direct_match(raw_value, candidates) if USE_DIRECT_MATCH else None

                if direct is not None:
                    answer = direct
                elif USE_LLM:
                    prompt = build_prompt(
                        question_id=question_id,
                        question_data=question_data,
                        raw_value=raw_value,
                        yaml_context=yaml_context,
                        pdf_context=pdf_context,
                        candidates=candidates,
                    )

                    answer = call_ollama(prompt, debug_name)
                    answer = validate_answer(answer, candidates)
                else:
                    answer = empty_answer("No direct match and LLM disabled.")

            except Exception as exc:
                answer = empty_answer(str(exc))

            output["answers"][section_name][question_id] = {
                "question": question_data.get("question"),
                "variable": question_data.get("variable"),
                "yaml_path": question_data.get("yaml_path"),
                "raw_value": raw_value,
                "selected_iri": answer.get("selected_iri"),
                "selected_name": answer.get("selected_name"),
                "selected_individual": answer.get("selected_individual"),
                "confidence": answer.get("confidence"),
                "source": answer.get("source"),
                "evidence": answer.get("evidence"),
                "reason": answer.get("reason"),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }

    write_json(OUTPUT_FILE, output)

    print(f"Written {OUTPUT_FILE}")
    print(f"Individuals loaded: {len(individuals)}")


run_extraction()