import json
import yaml
import requests
from pypdf import PdfReader
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent

YAML_FILE = BASE_DIR / "mattersim-v1-5M.yml"
PROMPT_FILE = BASE_DIR / "prompts.json"
ONTOLOGY_CONTEXT_FILE = BASE_DIR / "ontology_context.json"
PDF_FILE = BASE_DIR / "2405.04967v2.pdf"

OUTPUT_FILE = BASE_DIR / "outputs" / "model_extraction.json"
DEBUG_DIR = BASE_DIR / "debug"

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Target we can find in the yaml files
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

def build_prompt(yaml_data, ontology_context, prompts):
    yaml_summary = build_yaml_summary(yaml_data)

    return prompts["extraction_prompt"].format(
        yaml_content=json.dumps(yaml_summary, indent=2),
        ontology_context=json.dumps(ontology_context, indent=2),
        target_entities=json.dumps(TARGET_ENTITY_TYPES, indent=2),
        output_schema=json.dumps(EXPECTED_OUTPUT_SCHEMA, indent=2)
    )

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

yaml_data = load_yaml(YAML_FILE)
ontology_context = load_json(ONTOLOGY_CONTEXT_FILE)
prompts = load_json(PROMPT_FILE)

def query_ollama(prompt, json_format=True):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    if json_format:
        payload["format"] = "json"

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]

def build_yaml_summary(yaml_data):
    return {
        "model_name": yaml_data.get("model_name"),
        "model_key": yaml_data.get("model_key"),
        "model_type": yaml_data.get("model_type"),
        "model_params": yaml_data.get("model_params"),
        "notes_description": yaml_data.get("notes", {}).get("Description")
    }

def clean_llm_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text

def extract_yaml_facts(yaml_data):
    return {
        "model_name": yaml_data.get("model_name"),
        "model_key": yaml_data.get("model_key"),
        "model_type": yaml_data.get("model_type"),
        "model_params": yaml_data.get("model_params"),
        "description": yaml_data.get("notes", {}).get("Description"),
        "training_set": yaml_data.get("training_set"),
        "checkpoint_url": yaml_data.get("checkpoint_url")
    }

pdf_pages = load_pdf_text(PDF_FILE)

def get_by_path(data, path):
    current = data

    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def get_questions(prompts, yaml_data):
    questions = []

    for group in prompts["questionnaire"].values():
        for question_id, question_data in group.items():
            yaml_path = question_data.get("yaml_path")
            questions.append({
                "id": question_id,
                "question": question_data["question"],
                "variable": question_data["variable"],
                "source": question_data.get("source", "yaml"),
                "loop_over": question_data.get("loop_over"),
                "yaml_path": yaml_path,
                "yaml_value": get_by_path(yaml_data, yaml_path) if yaml_path else None
            })

    return questions

def build_question_prompt(question, prompts):
    return prompts["question_prompt"].format(
        question=question["question"],
        variable=question["variable"],
        yaml_value=json.dumps(question["yaml_value"], indent=2)
    )

def build_pdf_question_prompt(question, pdf_text, prompts):
    return prompts["pdf_question_prompt"].format(
        question=question["question"],
        variable=question["variable"],
        pdf_text=pdf_text
    )

def build_ontology_mapping_prompt(item, ontology_context, prompts):
    return prompts["ontology_mapping_prompt"].format(
        item=item,
        ontology_context=json.dumps(ontology_context, indent=2)
    )

def extract_answer_list(result):
    answer = result.get("answer")

    if isinstance(answer, list):
        return answer

    if isinstance(answer, str):
        return [item.strip() for item in answer.split(",")]

    return []

questions = get_questions(prompts, yaml_data)

def build_pdf_context(pdf_pages):
    return "\n\n".join(page["text"] for page in pdf_pages[:30])


pdf_text = build_pdf_context(pdf_pages)
results = {}

for question in questions:
    if question["source"] == "ontology_mapping":
        loop_values = extract_answer_list(results[question["loop_over"]])
        loop_results = []

        for value in loop_values:
            prompt = build_ontology_mapping_prompt(
                value,
                ontology_context,
                prompts
            )

            response = query_ollama(prompt)
            result = json.loads(clean_llm_json(response))
            loop_results.append(result)

        results[question["variable"]] = loop_results
        continue

    if question.get("loop_over"):
        loop_values = extract_answer_list(results[question["loop_over"]])
        loop_results = []

        for value in loop_values:
            loop_question = question.copy()
            loop_question["question"] = question["question"].format(item=value)

            prompt = build_pdf_question_prompt(loop_question, pdf_text, prompts)
            response = query_ollama(prompt)
            result = json.loads(clean_llm_json(response))

            loop_results.append({
                "item": value,
                "result": result
            })

        results[question["variable"]] = loop_results
        continue

    if question["source"] == "pdf" or question["yaml_value"] is None:
        prompt = build_pdf_question_prompt(question, pdf_text, prompts)
    else:
        prompt = build_question_prompt(question, prompts)

    response = query_ollama(prompt)
    result = json.loads(clean_llm_json(response))

    results[question["variable"]] = result

def build_rdf_prompt(results, ontology_context, prompts):
    return prompts["rdf_generation_prompt"].format(
        ontology_context=json.dumps(ontology_context, indent=2),
        prefixes=json.dumps(prompts["prefixes"], indent=2),
        results=json.dumps(results, indent=2)
    )

def save_ttl(ttl_content, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(ttl_content)
        
print(json.dumps(results, indent=2, ensure_ascii=False))

rdf_prompt = build_rdf_prompt(results, ontology_context, prompts)
ttl = query_ollama(rdf_prompt, json_format=False)
save_ttl(ttl, BASE_DIR / "outputs" / "model_llm.ttl")
