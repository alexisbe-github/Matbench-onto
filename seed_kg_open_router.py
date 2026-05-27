import json
import yaml
from pypdf import PdfReader
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from openrouter import OpenRouter, errors


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

ONTOLOGY_FILES = [
    BASE_DIR / "ontology/architecture.ttl",
    BASE_DIR / "ontology/trainingonto.ttl",
    BASE_DIR / "ontology/datasetonto.ttl"
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openrouter/owl-alpha"
#OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

YAML_FILE = Path(os.environ["YAML_FILE"])

PROMPT_FILE = BASE_DIR / "prompts.json"

PDF_FILE = Path(os.environ["PDF_FILE"])
PDF_URL = os.getenv("PDF_URL")

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

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in .env file")

CLIENT = OpenRouter(api_key=OPENROUTER_API_KEY)


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

            time.sleep(20)
            return response.choices[0].message.content

        except errors.TooManyRequestsResponseError as error:
            last_error = error
            wait_time = 30 * (attempt + 1)
            print(f"Rate limit OpenRouter. Retry in {wait_time}s...")
            time.sleep(wait_time)

    raise last_error


def build_combined_extraction_prompt(yaml_data, ontology_context, prompts, questions, pdf_text):
    yaml_summary = build_yaml_summary(yaml_data)

    return f"""
You must follow this nested prompts.json questionnaire exactly.

Questionnaire:
{json.dumps(questions, indent=2, ensure_ascii=False)}

YAML summary:
{json.dumps(yaml_summary, indent=2, ensure_ascii=False)}

Ontology context:
{json.dumps(ontology_context, indent=2, ensure_ascii=False)}

PDF context:
{pdf_text}

Return one valid nested JSON object.

Expected top-level keys:
- model
- architectures
- training

Rules:
- model must be an object.
- Do not return flat keys.
- Architectures must contain one object per architecture.
- if multiple architectures are mentioned, NEVER merge them.
- preserve all architectures found in the paper.
- each architecture object must contain its own fields.
- Use YAML values when they directly answer the question.
- Use PDF context when YAML is not sufficient.
- Use ontology_context only for ontology class mapping.
- Use null when unknown.
- Use "evidence" field to display the proof coming from the yaml, pdf or ontology file, with the section number or page number, for every value found
"""


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_ttl(ttl_content, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(ttl_content)


yaml_data = load_yaml(YAML_FILE)
ontology_context = load_ontology_context(ONTOLOGY_FILES)
prompts = load_json(PROMPT_FILE)
pdf_pages = load_pdf_text(PDF_FILE)
pdf_text = build_pdf_context(pdf_pages)
questions = get_questions(prompts, yaml_data)

extraction_prompt = build_combined_extraction_prompt(
    yaml_data,
    ontology_context,
    prompts,
    questions,
    pdf_text
)

response = query_openrouter(extraction_prompt)
raw_results = json.loads(clean_llm_json(response))

results = raw_results

print(json.dumps(results, indent=2, ensure_ascii=False))

results["_sources"] = {
    "pdf_file": str(PDF_FILE),
    "pdf_url": PDF_URL,
    "yaml_file": str(YAML_FILE)
}

save_json(results, OUTPUT_JSON_FILE)