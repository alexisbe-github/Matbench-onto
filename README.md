# Matbench Ontology Knowledge Graph

This project builds a knowledge graph around the machine-learning models and
benchmarks published by [Matbench Discovery](https://matbench-discovery.materialsproject.org/).
It combines model submission metadata, research papers, and benchmark results
with a set of OWL ontologies.

The repository can:

- scrape model pages, YAML submissions, and papers from Matbench Discovery;
- use an OpenRouter model to extract structured model and training metadata;
- convert the extracted JSON records to RDF/Turtle;
- validate generated graphs against local SHACL constraints;
- repair invalid model graphs with an LLM;
- generate RDF for Matbench datasets and benchmark tasks;
- upload the resulting named graphs to GraphDB;
- provide reusable SPARQL queries for inspecting the knowledge graph.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pipeline/` | Model scraping, LLM extraction, RDF conversion, validation, and repair |
| `metadata/` | Matbench dataset/task context and model release-date synchronization |
| `graphdb/` | GraphDB upload utilities |
| `ontology/` | Architecture, training, dataset, and evaluation ontologies and their base individuals |
| `model_yamls/` | Matbench Discovery model submission files |
| `papers/` | Research papers used as extraction context |
| `outputs/json/` | Structured model metadata extracted by the LLM |
| `outputs/ttl/` | Model individuals generated from the JSON extractions |
| `outputs/ttl_repaired/` | Validated or LLM-repaired model graphs |
| `outputs/shacl_reports*/` | Human-readable and RDF SHACL validation reports |
| `query/` | Documented SPARQL queries |

## Pipeline overview

```text
Matbench model page + YAML + paper
                 |
                 v
        OpenRouter extraction
                 |
                 v
        outputs/json/*.json
                 |
                 v
           JSON -> Turtle
                 |
                 v
         outputs/ttl/*.ttl
                 |
          SHACL validation
                 |
                 v
    outputs/ttl_repaired/*.ttl
                 |
                 v
              GraphDB
                 |
           SPARQL queries
```

## Requirements

- Python 3.10 or newer
- An [OpenRouter](https://openrouter.ai/) API key for LLM extraction and repair
- Internet access for scraping model pages and downloading source files
- GraphDB running locally if you want to load and query the complete graph

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root containing your
`OPENROUTER_API_KEY`. The default model is `meituan/longcat-2.0`; it can be
overridden with `OPENROUTER_MODEL`. The `.env` file is ignored by Git.

Run all commands below from the repository root.

## Quick start

Start with a single model to check the complete extraction path:

```bash
python pipeline/matbench_pipeline.py --limit 1
```

This downloads the model sources, calls OpenRouter, and writes both the JSON
extraction and its generated Turtle graph.

Validate the generated graph:

```bash
python pipeline/validate_shacl.py
```

## Model extraction and RDF generation

### Run the complete scraper

Process every model currently listed by Matbench Discovery:

```bash
python pipeline/matbench_pipeline.py
```

Process only a small batch:

```bash
python pipeline/matbench_pipeline.py --limit 3
```

Resume from a model slug, URL, YAML/PDF filename, JSON extraction, or TTL
filename:

```bash
python pipeline/matbench_pipeline.py --start-from chgnet_0_3_0
```

The pipeline stores downloaded inputs in `model_yamls/`, `papers/`, and
`outputs/model_pages/`. It writes extracted data to `outputs/json/` and model
individuals to `outputs/ttl/`. Errors are reported per model so that the rest
of the batch can continue.

### Convert existing JSON files without calling an LLM

Convert every extraction in `outputs/json/`:

```bash
python pipeline/convert_all_json_to_ttl.py
```

Convert one file by setting its input and output paths.

PowerShell:

```powershell
$env:INPUT_JSON_FILE = "outputs/json/chgnet_0_3_0_model_extraction.json"
$env:OUTPUT_TTL_FILE = "outputs/ttl/chgnet_0_3_0_model_individuals_generated.ttl"
python pipeline/json_to_ttl.py
```

Bash:

```bash
INPUT_JSON_FILE=outputs/json/chgnet_0_3_0_model_extraction.json \
OUTPUT_TTL_FILE=outputs/ttl/chgnet_0_3_0_model_individuals_generated.ttl \
python pipeline/json_to_ttl.py
```

### Run one custom extraction

`pipeline/seed_kg_open_router.py` expects its source paths through environment
variables. At minimum, set `YAML_FILE`, `PDF_FILE`, and `OUTPUT_JSON_FILE`.
`PDF_URL`, `MODEL_PAGE_FILE`, and `MODEL_PAGE_URL` add provenance and page
context when available.

PowerShell example:

```powershell
$env:YAML_FILE = "model_yamls/chgnet_0_3_0.yml"
$env:PDF_FILE = "papers/chgnet_0_3_0.pdf"
$env:OUTPUT_JSON_FILE = "outputs/json/chgnet_0_3_0_model_extraction.json"
python pipeline/seed_kg_open_router.py
```

## SHACL validation and repair

Validate all raw generated model graphs:

```bash
python pipeline/validate_shacl.py
```

Validate one graph:

```bash
python pipeline/validate_shacl.py --ttl outputs/ttl/chgnet_0_3_0_model_individuals_generated.ttl
```

Validate another directory and choose the report directory:

```bash
python pipeline/validate_shacl.py \
  --ttl-dir outputs/ttl_repaired \
  --report-dir outputs/shacl_reports_after_repair
```

Validation checks both the local SHACL constraints and unknown ontology classes
used as `rdf:type`. Reports are written as `.txt` and `.ttl`.

To run the LLM repair pass over the files in `outputs/ttl/`:

```bash
python pipeline/repair_ttl_with_llm.py
```

The repair script uses the corresponding JSON, YAML, paper, and validation
report as context. Repaired graphs are written to `outputs/ttl_repaired/`.
Because this operation makes OpenRouter calls, test extraction and validation
on a small selection before repairing a large batch.

## Metadata enrichment

Generate RDF individuals for the datasets and benchmark tasks published on the
Matbench Discovery site:

```bash
python metadata/generate_matbench_site_context_ttl.py
```

Synchronize model release dates from a checkout of the official
`matbench-discovery` repository:

```bash
python metadata/sync_matbench_release_dates.py \
  --source-root ../matbench-discovery
```

## Loading the graph into GraphDB

The upload scripts expect GraphDB at `http://localhost:7200` and a repository
named `matbench`. Both values can be overridden on the command line.

Before uploading generated individuals, create the repository and load the
ontology files from `ontology/` into stable named graphs. The model uploader
preserves these four graph names:

| Ontology | Named graph |
| --- | --- |
| Architecture | `https://k.loria.fr/graphs/ontology/architecture` |
| Dataset | `https://k.loria.fr/graphs/ontology/dataset` |
| Evaluation | `https://k.loria.fr/graphs/ontology/evaluation` |
| Training | `https://k.loria.fr/graphs/ontology/training` |

Upload every repaired model graph:

```bash
python graphdb/upload_to_graphdb.py
```

By default this clears all named graphs except the four ontology graphs above.
Use `--no-clear` to preserve existing graphs, or upload only one file:

```bash
python graphdb/upload_to_graphdb.py --no-clear
python graphdb/upload_to_graphdb.py \
  --ttl-file outputs/ttl_repaired/chgnet_0_3_0_model_individuals_generated.ttl
```

For a different server or repository:

```bash
python graphdb/upload_to_graphdb.py \
  --graphdb-url http://localhost:7200 \
  --repository matbench
```

## SPARQL queries

See [`query/README.md`](query/README.md) for the principal SPARQL queries used
to inspect model performance, parameter counts, release dates, and training
datasets in GraphDB.

## Important notes

- LLM extraction is evidence-assisted but not guaranteed to be correct. Keep
  the JSON extraction, Turtle conversion, and SHACL validation as separate
  reviewable stages.
- `graphdb/upload_to_graphdb.py` clears non-ontology named graphs by default. Use
  `--no-clear` when replacing the existing repository contents is not intended.
- Research papers make the repository relatively large; use a full clone when
  those local source files are required.
