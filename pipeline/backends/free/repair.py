from pathlib import Path
import json
import os
import re
import time

from dotenv import load_dotenv
from openrouter import OpenRouter, errors
from pypdf import PdfReader
from rdflib import Graph, Namespace, RDF
from rdflib.util import from_n3

from pipeline.validate_shacl import get_shacl_report


BASE_DIR = Path(__file__).resolve().parents[3]

TTL_DIR = BASE_DIR / "outputs" / "free_llm" / "ttl"
REPAIRED_TTL_DIR = BASE_DIR / "outputs" / "free_llm" / "ttl_repaired"

JSON_DIR = BASE_DIR / "outputs" / "free_llm" / "json"
YAML_DIR = BASE_DIR / "model_yamls"
PAPERS_DIR = BASE_DIR / "papers"
PAPER_TEXT_DIR = BASE_DIR / "outputs" / "paper_text"


load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "FREE_OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
PAPER_TEXT_MAX_CHARS = int(os.getenv("PAPER_TEXT_MAX_CHARS", "60000"))
INCLUDE_FALLBACK_SOURCES = os.getenv(
    "REPAIR_INCLUDE_FALLBACK_SOURCES",
    "0",
).lower() in {"1", "true", "yes"}

SH = Namespace("http://www.w3.org/ns/shacl#")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in .env file")

CLIENT = OpenRouter(api_key=OPENROUTER_API_KEY)


def read_text(path):
    path = Path(path)

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path):
    path = Path(path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_pdf_text(path, max_chars=PAPER_TEXT_MAX_CHARS):
    path = Path(path)

    if not path.exists():
        return ""

    pages = []
    total_chars = 0
    reader = PdfReader(path)

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        page_text = f"[page {index + 1}]\n{text}"
        remaining_chars = max_chars - total_chars

        if remaining_chars <= 0:
            break

        if len(page_text) > remaining_chars:
            page_text = page_text[:remaining_chars]

        pages.append(page_text)
        total_chars += len(page_text)

    return "\n\n".join(pages)


def strip_rdf_star_lines(ttl_text):
    cleaned_lines = []

    for line in ttl_text.splitlines():
        stripped = line.lstrip()

        if stripped.startswith("<<"):
            continue

        if ">>" in stripped and "prov:wasDerivedFrom" in stripped:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_rdf_star_lines(ttl_text):
    return [
        line
        for line in ttl_text.splitlines()
        if line.lstrip().startswith("<<")
    ]


def parse_rdf_star_quoted_triple(line):
    match = re.match(
        r"\s*<<\s*(<[^>]+>)\s+(<[^>]+>)\s+(.+?)\s*>>\s+prov:wasDerivedFrom",
        line,
    )

    if not match:
        return None

    subject_text, predicate_text, object_text = match.groups()

    try:
        return (
            from_n3(subject_text),
            from_n3(predicate_text),
            from_n3(object_text),
        )
    except Exception:
        return None


def restore_rdf_star_lines(repaired_ttl, original_ttl):
    rdf_star_lines = extract_rdf_star_lines(original_ttl)

    if not rdf_star_lines:
        return repaired_ttl.strip()

    cleaned_repaired_ttl = strip_rdf_star_lines(repaired_ttl).strip()

    graph = Graph()
    graph.parse(data=cleaned_repaired_ttl, format="turtle")

    kept_lines = []

    for line in rdf_star_lines:
        quoted_triple = parse_rdf_star_quoted_triple(line)

        if quoted_triple is None:
            continue

        if quoted_triple in graph:
            kept_lines.append(line)

    if not kept_lines:
        return cleaned_repaired_ttl

    return cleaned_repaired_ttl + "\n\n" + "\n".join(kept_lines)


def normalize_local_source_paths(ttl_text):
    normalized = ttl_text
    base_path = str(BASE_DIR)
    escaped_base_path = base_path.replace("\\", "\\\\")

    replacements = {
        base_path + "\\": "",
        base_path + "/": "",
        escaped_base_path + "\\\\": "",
        escaped_base_path + "/": "",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = normalized.replace("model_yamls\\\\", "model_yamls/")
    normalized = normalized.replace("papers\\\\", "papers/")

    return normalized


def source_url_from_model_key(model_key):
    json_path = JSON_DIR / f"{model_key}_model_extraction.json"
    data = load_json(json_path)
    sources = data.get("_sources", {})

    return sources.get("pdf_url")


def normalize_source_provenance(ttl_text, model_key):
    source_url = source_url_from_model_key(model_key)

    if not source_url:
        return normalize_local_source_paths(ttl_text)

    normalized = normalize_local_source_paths(ttl_text)

    return re.sub(
        r'prov:wasDerivedFrom\s+"[^"]*(?:model_yamls|papers)[^"]*"\s*(?:\^\^xsd:string)?',
        f"prov:wasDerivedFrom <{source_url}> ",
        normalized,
    )


def model_key_from_ttl_path(ttl_path):
    name = Path(ttl_path).stem
    return name.replace("_model_individuals_generated", "")


def find_context_files(model_key):
    json_path = JSON_DIR / f"{model_key}_model_extraction.json"
    yaml_path = YAML_DIR / f"{model_key}.yml"
    yaml_alt_path = YAML_DIR / f"{model_key}.yaml"
    paper_txt_path = PAPER_TEXT_DIR / f"{model_key}.txt"
    paper_pdf_path = PAPERS_DIR / f"{model_key}.pdf"

    return {
        "json": json_path if json_path.exists() else None,
        "yaml": yaml_path if yaml_path.exists() else (
            yaml_alt_path if yaml_alt_path.exists() else None
        ),
        "paper_text": paper_txt_path if paper_txt_path.exists() else None,
        "paper_pdf": paper_pdf_path if paper_pdf_path.exists() else None,
    }


def load_paper_context(model_key, files):
    if files["paper_text"]:
        return read_text(files["paper_text"])

    if not files["paper_pdf"]:
        return ""

    text = read_pdf_text(files["paper_pdf"])

    if text:
        cached_path = PAPER_TEXT_DIR / f"{model_key}.txt"
        write_text(cached_path, text)

    return text


def load_context(model_key):
    """Load concise repair evidence; JSON is authoritative for generated TTL."""
    files = find_context_files(model_key)
    parts = []

    if files["json"]:
        parts.append(
            "=== EXTRACTION JSON ===\n"
            + read_text(files["json"])
        )

    if INCLUDE_FALLBACK_SOURCES and files["yaml"]:
        parts.append(
            "=== MODEL YAML ===\n"
            + read_text(files["yaml"])
        )

    paper_context = (
        load_paper_context(model_key, files)
        if INCLUDE_FALLBACK_SOURCES
        else ""
    )

    if paper_context:
        parts.append(
            "=== PAPER TEXT ===\n"
            + paper_context
        )

    return "\n\n".join(parts)


def format_shacl_report(report):
    """Render only actionable fields, without pySHACL's repeated shape dumps."""
    report_graph = report["report_graph"]
    results = sorted(
        report_graph.subjects(RDF.type, SH.ValidationResult),
        key=str,
    )
    parts = ["=== ACTIONABLE SHACL VIOLATIONS ==="]

    for index, result in enumerate(results, start=1):
        focus = report_graph.value(result, SH.focusNode)
        path = report_graph.value(result, SH.resultPath)
        value = report_graph.value(result, SH.value)
        component = report_graph.value(result, SH.sourceConstraintComponent)
        messages = sorted({
            str(message)
            for message in report_graph.objects(result, SH.resultMessage)
        })

        parts.extend([
            f"[{index}] focusNode: {focus}",
            f"    resultPath: {path or '(node constraint)'}",
            f"    constraint: {component}",
        ])
        if value is not None:
            parts.append(f"    offendingValue: {value}")
        for message in messages:
            parts.append(f"    message: {message}")

    if report.get("unknown_types"):
        parts.extend([
            "",
            "=== UNKNOWN RDF TYPES ===",
            "\n".join(f"- {rdf_type}" for rdf_type in report["unknown_types"]),
        ])

    return "\n".join(parts)


def build_repair_prompt(ttl_text, shacl_report, source_context):
    return f"""
You are repairing RDF/Turtle generated by a LLM.

Your task is to repair ONLY the SHACL violations.

Rules:
- Return valid Turtle only.
- Do not explain your reasoning.
- Do not use local absolute filesystem paths.
- For model-level provenance, use the scraped paper URL as prov:wasDerivedFrom, not a local YAML/PDF path.
- Use dcterms:source for paper URLs on triples with textual evidence.
- Fix only triples directly related to SHACL violations.
- Prefer fixing wrong rdf:type statements and missing required links over inventing new entities.
- Use the extraction JSON as the evidence for repairs.
- A missing JSON value means unknown; never invent a numeric metric value.
- If a MetricResult has no numeric value in the JSON, remove that empty MetricResult and its link instead of fabricating a value.
- If required evidence is absent, remove an incomplete optional result or link; do not copy a value from another task or invent details.

=== SHACL REPORT ===
{shacl_report}

=== SOURCE CONTEXT ===
{source_context}

=== ORIGINAL TTL ===
{ttl_text}
""".strip()


def call_llm(prompt, max_retries=6):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an RDF/Turtle repair assistant. "
                "Return only valid Turtle. "
                "No prose, no markdown, no explanations. "
                "The first characters of your answer must be @prefix. "
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    last_error = None

    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.send(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0,
                stream=False,
            )

            content = response.choices[0].message.content

            if content is None:
                raise ValueError("OpenRouter returned empty content.")

            time.sleep(20)
            return content

        except errors.TooManyRequestsResponseError as error:
            last_error = error
            wait_time = 30 * (attempt + 1)
            print(f"Rate limit OpenRouter. Retry in {wait_time}s...")
            time.sleep(wait_time)

    raise last_error


def extract_turtle_from_text(text):
    fenced_match = re.search(
        r"```(?:ttl|turtle)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        return fenced_match.group(1).strip()

    prefix_index = text.find("@prefix")

    if prefix_index != -1:
        return text[prefix_index:].strip()

    base_index = text.find("@base")

    if base_index != -1:
        return text[base_index:].strip()

    return text.strip()


def clean_llm_ttl_output(text):
    if text is None:
        raise ValueError("LLM returned None instead of Turtle text.")

    text = extract_turtle_from_text(text)
    text = strip_rdf_star_lines(text)

    return text


def validate_turtle_syntax(ttl_text):
    graph = Graph()
    graph.parse(data=ttl_text, format="turtle")
    return True


def turtle_triple_count(ttl_text):
    graph = Graph()
    graph.parse(data=strip_rdf_star_lines(ttl_text), format="turtle")
    return len(graph)


def is_suspiciously_short_repair(original_ttl, repaired_ttl):
    original_count = turtle_triple_count(original_ttl)
    repaired_count = turtle_triple_count(repaired_ttl)

    if original_count < 20:
        return False

    minimum_expected_count = max(10, int(original_count * 0.80))
    return repaired_count < minimum_expected_count


def build_syntax_retry_prompt(original_prompt, bad_ttl, syntax_error):
    return f"""
Your previous answer was not valid Turtle.

Return the complete repaired Turtle document again.

Hard rules:
- Return Turtle only.
- No prose.
- No markdown fences.
- Start with @prefix.
- Include all prefixes needed by the document.
- Do not return a partial diff.

=== TURTLE SYNTAX ERROR ===
{syntax_error}

=== INVALID ANSWER ===
{bad_ttl}

=== ORIGINAL REPAIR TASK ===
{original_prompt}
""".strip()


def repair_one_file(ttl_path):
    ttl_path = Path(ttl_path)
    model_key = model_key_from_ttl_path(ttl_path)

    original_ttl = read_text(ttl_path)
    current_ttl = strip_rdf_star_lines(original_ttl)
    source_context = load_context(model_key)

    output_path = REPAIRED_TTL_DIR / ttl_path.name
    temporary_path = REPAIRED_TTL_DIR / f"__tmp_{ttl_path.name}"

    write_text(temporary_path, current_ttl)

    report = get_shacl_report(temporary_path)

    if report["conforms"]:
        write_text(output_path, original_ttl)
        temporary_path.unlink(missing_ok=True)
        print(f"[OK] {ttl_path.name} already conforms")
        return True

    print(f"[REPAIR] {ttl_path.name}")

    prompt = build_repair_prompt(
        ttl_text=current_ttl,
        shacl_report=format_shacl_report(report),
        source_context=source_context,
    )

    repaired_ttl = clean_llm_ttl_output(call_llm(prompt))
    syntax_error = None

    for syntax_attempt in range(2):
        try:
            validate_turtle_syntax(repaired_ttl)
            syntax_error = None
            break
        except Exception as error:
            syntax_error = error

            if syntax_attempt == 1:
                break

            print(f"[RETRY] LLM returned invalid Turtle for {ttl_path.name}")
            retry_prompt = build_syntax_retry_prompt(
                original_prompt=prompt,
                bad_ttl=repaired_ttl,
                syntax_error=error,
            )
            repaired_ttl = clean_llm_ttl_output(call_llm(retry_prompt))

    if syntax_error:
        debug_path = REPAIRED_TTL_DIR / f"__invalid_{ttl_path.name}"
        write_text(debug_path, repaired_ttl)

        print(f"[FAIL] LLM returned invalid Turtle for {ttl_path.name}")
        print(syntax_error)
        print(f"[DEBUG] Invalid TTL written to {debug_path}")

        temporary_path.unlink(missing_ok=True)
        return False

    if is_suspiciously_short_repair(current_ttl, repaired_ttl):
        debug_path = REPAIRED_TTL_DIR / f"__truncated_{ttl_path.name}"
        write_text(debug_path, repaired_ttl)

        print(f"[FAIL] repaired TTL looks truncated: {ttl_path.name}")
        print(f"[DEBUG] Truncated candidate written to {debug_path}")

        temporary_path.unlink(missing_ok=True)
        return False

    repaired_ttl = normalize_source_provenance(repaired_ttl, model_key)
    repaired_ttl_with_rdf_star = restore_rdf_star_lines(
        repaired_ttl,
        original_ttl,
    )

    write_text(output_path, repaired_ttl_with_rdf_star)

    final_tmp_path = REPAIRED_TTL_DIR / f"__final_{ttl_path.name}"
    write_text(final_tmp_path, repaired_ttl_with_rdf_star)

    final_report = get_shacl_report(final_tmp_path)

    temporary_path.unlink(missing_ok=True)
    final_tmp_path.unlink(missing_ok=True)

    if final_report["conforms"]:
        print(f"[OK] repaired {ttl_path.name}")
        return True

    print(f"[FAIL] repaired TTL still violates SHACL: {ttl_path.name}")
    print(final_report["report_text"])
    return False


def main():
    REPAIRED_TTL_DIR.mkdir(parents=True, exist_ok=True)

    ttl_files = sorted(TTL_DIR.glob("*.ttl"))

    if not ttl_files:
        print(f"No TTL files found in {TTL_DIR}")

    failed = []

    for ttl_path in ttl_files:
        try:
            ok = repair_one_file(ttl_path)
            if not ok:
                failed.append(ttl_path)
        except Exception as error:
            print(f"[ERROR] {ttl_path.name}")
            print(error)
            failed.append(ttl_path)

    if failed:
        print(f"\nRepair failed for {len(failed)} file(s).")

    print(f"\nRepaired or validated {len(ttl_files)} file(s).")


if __name__ == "__main__":
    main()
