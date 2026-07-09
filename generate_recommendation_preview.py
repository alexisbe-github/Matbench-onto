import argparse
import html
import json
import os
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "recommendation_platform_preview"


def slug_from_json(path):
    suffix = "_model_extraction"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def slug_from_ttl(path):
    suffix = "_model_individuals_generated"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def escape(value):
    return html.escape(str(value), quote=True)


def value_of(node, default=None):
    if isinstance(node, dict) and "value" in node:
        return node.get("value")
    return node if node is not None else default


def text_value(node, default="N/A"):
    value = value_of(node, default)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def page_path(output_dir, slug):
    return output_dir / "models" / f"{slug}.html"


def rel(from_path, to_path):
    return Path(os.path.relpath(Path(to_path).resolve(), Path(from_path).resolve().parent)).as_posix()


def load_json(path):
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path):
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def collect_prediction_manifest(path):
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_model = {}
    for entry in manifest:
        by_model.setdefault(entry.get("model_slug"), []).append(entry)
    return by_model


def iter_prediction_entries(value):
    if isinstance(value, dict):
        if value.get("pred_file") and value.get("pred_file_url"):
            yield {
                "pred_file": value["pred_file"],
                "pred_file_url": value["pred_file_url"],
                "filename": Path(str(value["pred_file"])).name,
            }
        for child in value.values():
            yield from iter_prediction_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_prediction_entries(child)


def prediction_entries_from_yaml(slug, yaml_data, manifest_entries):
    by_filename = {
        entry.get("filename"): entry for entry in manifest_entries if entry.get("filename")
    }
    entries = []
    for entry in iter_prediction_entries(yaml_data):
        filename = entry["filename"]
        local_path = BASE_DIR / "outputs" / "pred_files" / slug / filename
        manifest_entry = by_filename.get(filename, {})
        status = manifest_entry.get("status")
        size = manifest_entry.get("size_bytes")
        if local_path.exists() and local_path.stat().st_size > 0:
            status = "local"
            size = local_path.stat().st_size
        elif not status:
            status = "missing"
        entries.append(
            {
                **entry,
                "status": status,
                "size_bytes": size,
                "output_path": str(local_path.relative_to(BASE_DIR)),
            }
        )
    return entries


def render_scalar(value):
    if value is None:
        return '<span class="muted">null</span>'
    if isinstance(value, bool):
        return f'<span class="bool">{str(value).lower()}</span>'
    if isinstance(value, (int, float)):
        return f'<span class="number">{escape(value)}</span>'
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return f'<a href="{escape(value)}">{escape(value)}</a>'
    return escape(value)


def render_node(node):
    if isinstance(node, dict):
        if set(node.keys()).issubset({"value", "evidence", "source", "ontology_class"}):
            pieces = [f'<div class="value">{render_node(node.get("value"))}</div>']
            if node.get("ontology_class"):
                pieces.append(
                    f'<div class="meta">class: {render_node(node.get("ontology_class"))}</div>'
                )
            if node.get("evidence"):
                pieces.append(f'<div class="evidence">{escape(node["evidence"])}</div>')
            if node.get("source"):
                pieces.append(f'<div class="source">source: {render_scalar(node["source"])}</div>')
            return "".join(pieces)

        rows = []
        for key, value in node.items():
            rows.append(
                "<tr>"
                f"<th>{escape(key)}</th>"
                f"<td>{render_node(value)}</td>"
                "</tr>"
            )
        return f'<table class="attrs">{"".join(rows)}</table>'

    if isinstance(node, list):
        if not node:
            return '<span class="muted">[]</span>'
        items = "".join(f"<li>{render_node(item)}</li>" for item in node)
        return f"<ul>{items}</ul>"

    return render_scalar(node)


def render_section(title, node):
    if not node:
        return ""
    return (
        '<section class="band">'
        f"<h2>{escape(title)}</h2>"
        f"{render_node(node)}"
        "</section>"
    )


def render_simple_table(headers, rows, empty_message):
    if not rows:
        return f'<p class="muted">{escape(empty_message)}</p>'
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{cell}</td>" for cell in row)
            + "</tr>"
        )
    return f'<table class="attrs compact"><tr>{head}</tr>{"".join(body)}</table>'


def render_metric_badges(metrics):
    if not metrics:
        return '<span class="muted">N/A</span>'
    badges = []
    for key, value in metrics.items():
        badges.append(
            f'<span class="metric"><strong>{escape(key)}</strong> {render_scalar(value)}</span>'
        )
    return f'<div class="metric-list">{"".join(badges)}</div>'


def is_metric_scalar(value):
    return isinstance(value, (int, float, str, bool)) and value is not None


METADATA_KEYS = {
    "pred_file",
    "pred_file_url",
    "pred_col",
    "struct_col",
    "analysis_file",
    "analysis_file_url",
}


def split_metric_leaf(node):
    if not isinstance(node, dict):
        return {}, {}
    metrics = {
        key: value
        for key, value in node.items()
        if key not in METADATA_KEYS and is_metric_scalar(value)
    }
    metadata = {key: value for key, value in node.items() if key in METADATA_KEYS}
    return metrics, metadata


def iter_metric_rows(task, node, path=()):
    if node == "not applicable":
        yield {
            "task": task,
            "scope": "not applicable",
            "metrics": {"status": "not applicable"},
            "metadata": {},
        }
        return
    if not isinstance(node, dict):
        return

    metrics, metadata = split_metric_leaf(node)
    child_items = [
        (key, value)
        for key, value in node.items()
        if key not in METADATA_KEYS and isinstance(value, dict)
    ]
    if metrics:
        yield {
            "task": task,
            "scope": " / ".join(path) if path else "global",
            "metrics": metrics,
            "metadata": metadata,
        }
    for key, value in child_items:
        child_metrics, _ = split_metric_leaf(value)
        if child_metrics:
            yield from iter_metric_rows(task, value, (*path, str(key)))
        else:
            yield from iter_metric_rows(task, value, (*path, str(key)))


def collect_metric_rows(yaml_data):
    rows = []
    for task, node in (yaml_data.get("metrics") or {}).items():
        rows.extend(iter_metric_rows(task, node))
    return rows


def render_metrics_section(record):
    metric_rows = collect_metric_rows(record["yaml"])
    rows = []
    for row in metric_rows:
        rows.append(
            [
                escape(row["task"]),
                escape(row["scope"]),
                render_metric_badges(row["metrics"]),
            ]
        )
    return (
        '<section class="band">'
        "<h2>Métriques et résultats</h2>"
        + render_simple_table(
            ["Task", "Jeu / split", "Métriques"],
            rows,
            "Aucune métrique structurée trouvée dans le YAML.",
        )
        + "</section>"
    )


def render_tasks_section(record):
    yaml_data = record["yaml"]
    notes = yaml_data.get("notes") or {}
    rows = []
    for key, label in [
        ("train_task", "Training task"),
        ("test_task", "Test task"),
        ("targets", "Targets"),
        ("model_type", "Model type"),
        ("openness", "Openness"),
    ]:
        if key in yaml_data:
            rows.append([escape(label), render_node(yaml_data[key])])

    metric_tasks = []
    for task, value in (yaml_data.get("metrics") or {}).items():
        status = value if isinstance(value, str) else "available"
        metric_tasks.append(f"{task} ({status})")
    if metric_tasks:
        rows.append(["Benchmark tasks", render_node(metric_tasks)])
    if notes.get("Steps"):
        rows.append(["Training steps", render_node(notes["Steps"])])

    return (
        '<section class="band">'
        "<h2>Tasks et protocole</h2>"
        + render_simple_table(
            ["Champ", "Valeur"],
            rows,
            "Aucune task structurée trouvée.",
        )
        + "</section>"
    )


def unique_rows(rows):
    seen = set()
    unique = []
    for row in rows:
        key = tuple(str(cell) for cell in row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def dataset_names_from_node(node):
    value = value_of(node)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        names = []
        for item in value:
            names.extend(dataset_names_from_node(item))
        return names
    return []


def render_datasets_section(record):
    rows = []
    yaml_data = record["yaml"]
    for dataset in dataset_names_from_node(yaml_data.get("training_set")):
        rows.append([escape(dataset), "YAML training_set"])

    architectures = record["json"].get("architectures") or {}
    arch_items = architectures.items() if isinstance(architectures, dict) else []
    for arch_name, arch in arch_items:
        if not isinstance(arch, dict):
            continue
        for dataset in dataset_names_from_node(arch.get("datasets")):
            rows.append([escape(dataset), f"Architecture: {escape(arch_name)}"])

    training_details = (record["json"].get("training") or {}).get("training_run_details") or {}
    if isinstance(training_details, dict):
        for run_name, run in training_details.items():
            if isinstance(run, dict):
                for dataset in dataset_names_from_node(run.get("datasets")):
                    rows.append([escape(dataset), f"Training run: {escape(run_name)}"])

    return (
        '<section class="band">'
        "<h2>Datasets</h2>"
        + render_simple_table(
            ["Dataset", "Contexte"],
            unique_rows(rows),
            "Aucun dataset structuré trouvé.",
        )
        + "</section>"
    )


def hyperparameter_rows_from_architectures(json_data):
    rows = []
    architectures = json_data.get("architectures") or {}
    arch_items = architectures.items() if isinstance(architectures, dict) else []
    for arch_name, arch in arch_items:
        if not isinstance(arch, dict):
            continue
        details = arch.get("hyperparameter_details")
        if isinstance(details, dict):
            for name, detail in details.items():
                rows.append(
                    [
                        f"Architecture: {escape(arch_name)}",
                        escape(name),
                        render_node(value_of(detail, detail)),
                    ]
                )
        hyperparams = arch.get("hyperparameters")
        if isinstance(hyperparams, list):
            for item in hyperparams:
                if isinstance(item, dict):
                    name = item.get("name") or text_value(item.get("value"), "hyperparameter")
                    value = item.get("value", item)
                    rows.append(
                        [
                            f"Architecture: {escape(arch_name)}",
                            escape(name),
                            render_node(value),
                        ]
                    )
                else:
                    rows.append(
                        [f"Architecture: {escape(arch_name)}", escape(item), ""]
                    )
    return rows


def hyperparameter_rows_from_training(json_data):
    rows = []
    details = (json_data.get("training") or {}).get("training_run_details") or {}
    if not isinstance(details, dict):
        return rows
    for run_name, run in details.items():
        if not isinstance(run, dict):
            continue
        hyperparams = run.get("hyperparameters") or run.get("hyperparameter_details")
        if isinstance(hyperparams, dict):
            for name, value in hyperparams.items():
                rows.append(
                    [f"Training run: {escape(run_name)}", escape(name), render_node(value)]
                )
        elif isinstance(hyperparams, list):
            for item in hyperparams:
                if isinstance(item, dict):
                    name = item.get("name") or text_value(item.get("value"), "hyperparameter")
                    rows.append(
                        [
                            f"Training run: {escape(run_name)}",
                            escape(name),
                            render_node(item.get("value", item)),
                        ]
                    )
                else:
                    rows.append([f"Training run: {escape(run_name)}", escape(item), ""])
    return rows


def render_hyperparameters_section(record):
    rows = hyperparameter_rows_from_architectures(record["json"])
    rows.extend(hyperparameter_rows_from_training(record["json"]))
    return (
        '<section class="band">'
        "<h2>Hyperparamètres</h2>"
        + render_simple_table(
            ["Contexte", "Hyperparamètre", "Valeur / preuve"],
            unique_rows(rows),
            "Aucun hyperparamètre structuré trouvé.",
        )
        + "</section>"
    )


def render_result_files_section(record):
    rows = []
    for entry in record["predictions"]:
        status = entry.get("status", "unknown")
        size = entry.get("size_bytes")
        size_label = f"{size / 1024 / 1024:.1f} MB" if isinstance(size, int) else "N/A"
        rows.append(
            [
                escape(entry.get("filename", "")),
                f"<span class='pill {escape(status)}'>{escape(status)}</span>",
                escape(size_label),
                render_scalar(entry.get("pred_file_url", "")),
                escape(entry.get("output_path") or ""),
            ]
        )
    return (
        '<section class="band">'
        "<h2>Fichiers de résultats / prédictions</h2>"
        + render_simple_table(
            ["Fichier", "Statut", "Taille", "URL", "Chemin local"],
            rows,
            "Aucun fichier de prédiction référencé dans les YAML.",
        )
        + "</section>"
    )


MODEL_INSPECTOR_SCRIPT = """
<script>
(() => {
  const panel = document.querySelector('#object-inspector');
  const title = panel?.querySelector('[data-inspector-title]');
  const kind = panel?.querySelector('[data-inspector-kind]');
  const body = panel?.querySelector('[data-inspector-body]');
  const source = panel?.querySelector('[data-inspector-source]');
  const close = panel?.querySelector('[data-inspector-close]');
  if (!panel || !title || !kind || !body || !source) return;

  const clean = (text) => (text || '').replace(/\\s+/g, ' ').trim();

  function describe(target) {
    const metric = target.closest('.metric');
    if (metric) {
      return {
        title: clean(metric.querySelector('strong')?.textContent) || 'Métrique',
        kind: 'metric',
        body: clean(metric.textContent),
        source: clean(metric.closest('tr')?.querySelector('td:nth-child(2)')?.textContent)
      };
    }

    const row = target.closest('tr');
    if (row) {
      const header = clean(row.querySelector(':scope > th')?.textContent)
        || clean(row.querySelector(':scope > td:first-child')?.textContent)
        || 'Objet';
      const section = clean(row.closest('section')?.querySelector('h2')?.textContent);
      const evidence = clean(row.querySelector('.evidence')?.textContent);
      const sourceText = clean(row.querySelector('.source')?.textContent)
        || clean(row.querySelector('a')?.getAttribute('href'));
      return {
        title: header,
        kind: section || 'attribut',
        body: clean(row.textContent),
        source: evidence || sourceText
      };
    }

    const item = target.closest('li, .facts span, h1, h2, a, .pill, .value');
    if (item) {
      return {
        title: clean(item.textContent) || clean(item.getAttribute('href')) || 'Objet',
        kind: clean(item.closest('section')?.querySelector('h2')?.textContent) || item.tagName.toLowerCase(),
        body: clean(item.textContent) || clean(item.getAttribute('href')),
        source: clean(item.getAttribute('href')) || clean(item.closest('tr')?.querySelector('.evidence')?.textContent)
      };
    }

    return null;
  }

  document.addEventListener('click', (event) => {
    const candidate = event.target.closest('tr, li, .metric, .facts span, h1, h2, a, .pill, .value');
    if (!candidate || candidate.closest('#object-inspector')) return;
    const info = describe(candidate);
    if (!info) return;

    if (candidate.closest('a') && !(event.ctrlKey || event.metaKey || event.shiftKey)) {
      event.preventDefault();
    }
    event.stopPropagation();

    title.textContent = info.title || 'Objet';
    kind.textContent = info.kind || 'objet';
    body.textContent = info.body || 'Pas de description disponible.';
    source.textContent = info.source || 'Source non renseignée pour cet objet.';
    panel.classList.add('open');
  });

  close?.addEventListener('click', () => panel.classList.remove('open'));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') panel.classList.remove('open');
  });
})();
</script>
"""


def render_object_inspector():
    return """
<aside id="object-inspector" class="inspector" aria-live="polite">
  <button type="button" class="inspector-close" data-inspector-close>×</button>
  <p data-inspector-kind>objet</p>
  <h2 data-inspector-title>Sélectionne un objet</h2>
  <div class="inspector-body" data-inspector-body>
    Clique sur une métrique, un dataset, une task, un hyperparamètre, un lien ou une ligne du tableau.
  </div>
  <div class="inspector-source" data-inspector-source>La description apparaîtra ici.</div>
</aside>
"""


def collect_model_records():
    yaml_paths = {path.stem: path for path in (BASE_DIR / "model_yamls").glob("*.yml")}
    json_paths = {slug_from_json(path): path for path in (BASE_DIR / "outputs" / "json").glob("*.json")}
    ttl_paths = {
        slug_from_ttl(path): path
        for path in (BASE_DIR / "outputs" / "ttl_repaired").glob("*.ttl")
        if "_generated_2" not in path.name and not path.name.startswith("__")
    }
    pred_manifest = collect_prediction_manifest(BASE_DIR / "outputs" / "pred_files" / "manifest.json")

    slugs = sorted(set(yaml_paths) | set(json_paths) | set(ttl_paths))
    records = []
    for slug in slugs:
        yaml_data = load_yaml(yaml_paths.get(slug))
        json_data = load_json(json_paths.get(slug))
        model = json_data.get("model", {})
        record = {
            "slug": slug,
            "yaml_path": yaml_paths.get(slug),
            "json_path": json_paths.get(slug),
            "ttl_path": ttl_paths.get(slug),
            "yaml": yaml_data,
            "json": json_data,
            "predictions": prediction_entries_from_yaml(
                slug, yaml_data, pred_manifest.get(slug, [])
            ),
            "name": text_value(model.get("variant"), yaml_data.get("model_name", slug)),
            "family": text_value(model.get("family"), yaml_data.get("model_name", slug).split("-")[0]),
            "params": text_value(model.get("parameter_number"), yaml_data.get("model_params")),
        }
        records.append(record)
    return records


def render_prediction_table(entries):
    if not entries:
        return '<p class="muted">Aucun fichier de prédiction référencé dans le manifeste local.</p>'
    rows = []
    for entry in entries:
        status = entry.get("status", "unknown")
        size = entry.get("size_bytes")
        size_label = f"{size / 1024 / 1024:.1f} MB" if isinstance(size, int) else "N/A"
        output_path = entry.get("output_path")
        rows.append(
            "<tr>"
            f"<td>{escape(entry.get('filename', ''))}</td>"
            f"<td><span class='pill {escape(status)}'>{escape(status)}</span></td>"
            f"<td>{escape(size_label)}</td>"
            f"<td>{render_scalar(entry.get('pred_file_url', ''))}</td>"
            f"<td>{escape(output_path or '')}</td>"
            "</tr>"
        )
    return (
        '<table class="attrs compact">'
        "<tr><th>Fichier</th><th>Statut</th><th>Taille</th><th>URL</th><th>Chemin local</th></tr>"
        f"{''.join(rows)}"
        "</table>"
    )


def render_model_page(record, output_dir):
    page = page_path(output_dir, record["slug"])
    index_href = rel(page, output_dir / "index.html")
    css_href = rel(page, output_dir / "assets" / "style.css")

    yaml_summary = {
        key: record["yaml"].get(key)
        for key in [
            "model_name",
            "model_key",
            "model_params",
            "date_added",
            "url",
            "paper",
            "pr_url",
            "checkpoint_url",
            "license",
        ]
        if key in record["yaml"]
    }
    files = {
        "yaml": str(record["yaml_path"].relative_to(BASE_DIR)) if record["yaml_path"] else None,
        "json": str(record["json_path"].relative_to(BASE_DIR)) if record["json_path"] else None,
        "ttl_repaired": str(record["ttl_path"].relative_to(BASE_DIR)) if record["ttl_path"] else None,
    }

    sections = [
        render_section("Résumé YAML", yaml_summary),
        render_section("Sources locales", files),
        render_section("Modèle", record["json"].get("model")),
        render_section("Architectures", record["json"].get("architectures")),
        render_section("Training", record["json"].get("training")),
        render_section("Évaluations", record["json"].get("evaluation")),
        render_section("Sources d'extraction", record["json"].get("_sources")),
        '<section class="band"><h2>Fichiers de prédiction</h2>'
        f'{render_prediction_table(record["predictions"])}</section>',
    ]
    sections = [
        render_section(
            "Identité du modèle",
            {**yaml_summary, **(record["json"].get("model") or {})},
        ),
        render_tasks_section(record),
        render_metrics_section(record),
        render_result_files_section(record),
        render_datasets_section(record),
        render_hyperparameters_section(record),
        render_section("Architecture", record["json"].get("architectures")),
        render_section("Training runs", record["json"].get("training")),
        render_section("Sources locales", files),
        render_section("Sources d'extraction", record["json"].get("_sources")),
    ]

    remaining = {
        key: value
        for key, value in record["json"].items()
        if key not in {"model", "architectures", "training", "evaluation", "_sources"}
    }
    if remaining:
        sections.append(render_section("Autres attributs JSON", remaining))

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(record["name"])} - preview</title>
  <link rel="stylesheet" href="{css_href}">
</head>
<body>
  <header class="topbar">
    <a href="{index_href}" class="back">Tous les modèles</a>
    <span>{escape(record["slug"])}</span>
  </header>
  <main>
    <section class="hero">
      <p>{escape(record["family"])}</p>
      <h1>{escape(record["name"])}</h1>
      <div class="facts">
        <span>{escape(record["params"])} paramètres</span>
        <span>{len(record["predictions"])} fichiers de prédiction</span>
        <span>{'TTL OK' if record["ttl_path"] else 'TTL absent'}</span>
      </div>
    </section>
    {''.join(sections)}
  </main>
  {render_object_inspector()}
  {MODEL_INSPECTOR_SCRIPT}
</body>
</html>
"""


def render_index(records, output_dir):
    css_href = "assets/style.css"
    cards = []
    for record in records:
        href = rel(output_dir / "index.html", page_path(output_dir, record["slug"]))
        status = "complete" if record["json_path"] and record["ttl_path"] else "partial"
        cards.append(
            '<article class="model-card" '
            f'data-name="{escape(record["name"].lower())}" '
            f'data-family="{escape(record["family"].lower())}">'
            f'<a href="{href}">'
            f'<span class="status {status}">{escape(status)}</span>'
            f'<h2>{escape(record["name"])}</h2>'
            f'<p>{escape(record["family"])}</p>'
            '<div class="facts">'
            f'<span>{escape(record["params"])} params</span>'
            f'<span>{len(record["predictions"])} pred</span>'
            f'<span>{"JSON" if record["json_path"] else "no JSON"}</span>'
            '</div>'
            '</a>'
            '</article>'
        )

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matbench Discovery - preview plateforme</title>
  <link rel="stylesheet" href="{css_href}">
</head>
<body>
  <main>
    <section class="hero">
      <p>Prévisualisation plateforme de recommandation</p>
      <h1>Matbench Discovery Models</h1>
      <div class="facts">
        <span>{len(records)} modèles</span>
        <span>{sum(1 for r in records if r["json_path"])} JSON</span>
        <span>{sum(1 for r in records if r["ttl_path"])} TTL repaired</span>
      </div>
      <input id="search" class="search" type="search" placeholder="Filtrer par nom ou famille">
    </section>
    <section class="grid" id="grid">
      {''.join(cards)}
    </section>
  </main>
  <script>
    const search = document.querySelector('#search');
    const cards = [...document.querySelectorAll('.model-card')];
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const haystack = `${{card.dataset.name}} ${{card.dataset.family}}`;
        card.hidden = query && !haystack.includes(query);
      }}
    }});
  </script>
</body>
</html>
"""


STYLE = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #687380;
  --line: #d9dee5;
  --accent: #0f766e;
  --accent-2: #b45309;
  --ok: #15803d;
  --warn: #b45309;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto 48px; }
.topbar {
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  color: var(--muted);
}
.back { font-weight: 650; }
.hero {
  padding: 34px 0 26px;
  border-bottom: 1px solid var(--line);
}
.hero p {
  margin: 0 0 6px;
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
  font-size: 12px;
}
h1 {
  margin: 0;
  font-size: clamp(32px, 6vw, 58px);
  line-height: 1;
  letter-spacing: 0;
}
h2 {
  margin: 0 0 14px;
  font-size: 18px;
  letter-spacing: 0;
}
.facts {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}
.facts span, .pill, .status {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 4px 9px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #eef2f6;
  color: #344151;
  font-size: 12px;
  font-weight: 650;
}
.search {
  margin-top: 22px;
  width: min(520px, 100%);
  height: 42px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 12px;
  font: inherit;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  padding-top: 22px;
}
.model-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.model-card a {
  display: block;
  min-height: 160px;
  padding: 16px;
  color: inherit;
}
.model-card h2 {
  margin-top: 20px;
  margin-bottom: 6px;
  font-size: 20px;
}
.model-card p {
  margin: 0;
  color: var(--muted);
}
.status.complete { color: var(--ok); background: #ecfdf3; border-color: #bbf7d0; }
.status.partial { color: var(--warn); background: #fff7ed; border-color: #fed7aa; }
.band {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow-x: auto;
}
.band tr,
.band li,
.band .metric,
.band .value,
.facts span,
.hero h1,
.hero p {
  cursor: pointer;
}
.band tr:hover > th,
.band tr:hover > td,
.band li:hover,
.band .metric:hover,
.facts span:hover {
  background: #f8fafc;
}
.attrs {
  width: 100%;
  border-collapse: collapse;
}
.attrs th, .attrs td {
  vertical-align: top;
  border-top: 1px solid var(--line);
  padding: 9px 10px;
}
.attrs tr:first-child > th,
.attrs tr:first-child > td {
  border-top: 0;
}
.attrs th {
  width: 210px;
  color: #344151;
  font-weight: 700;
  text-align: left;
}
.compact th { width: auto; }
ul { margin: 0; padding-left: 18px; }
li + li { margin-top: 4px; }
.value { font-weight: 600; }
.meta, .source, .evidence, .muted {
  color: var(--muted);
  font-size: 12px;
}
.evidence {
  margin-top: 4px;
  padding-left: 8px;
  border-left: 2px solid var(--line);
}
.number { color: var(--accent-2); font-weight: 700; }
.bool { color: var(--accent); font-weight: 700; }
.metric-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.metric {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  min-height: 26px;
  padding: 3px 8px;
  border: 1px solid #dbeafe;
  border-radius: 6px;
  background: #eff6ff;
  color: #1f3b64;
  white-space: nowrap;
}
.pill.downloaded, .pill.exists { color: var(--ok); background: #ecfdf3; border-color: #bbf7d0; }
.pill.local { color: var(--ok); background: #ecfdf3; border-color: #bbf7d0; }
.pill.failed { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
.pill.pending, .pill.missing { color: var(--warn); background: #fff7ed; border-color: #fed7aa; }
.inspector {
  position: fixed;
  top: 68px;
  right: 18px;
  z-index: 20;
  width: min(390px, calc(100vw - 28px));
  max-height: calc(100vh - 88px);
  overflow: auto;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: 0 18px 45px rgba(23, 32, 42, .16);
  transform: translateX(calc(100% + 28px));
  transition: transform .18s ease;
}
.inspector.open {
  transform: translateX(0);
}
.inspector-close {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 30px;
  height: 30px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f8fafc;
  color: var(--text);
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
}
.inspector p {
  margin: 0 36px 6px 0;
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
  font-size: 12px;
}
.inspector h2 {
  margin-right: 36px;
}
.inspector-body {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.inspector-source {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
@media (max-width: 700px) {
  main { width: min(100vw - 20px, 1180px); }
  .topbar { padding: 0 12px; }
  .attrs th, .attrs td { display: block; width: 100%; }
  .attrs td { border-top: 0; padding-top: 0; }
  .inspector {
    top: auto;
    right: 10px;
    bottom: 10px;
    max-height: 58vh;
  }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    records = collect_model_records()
    (output_dir / "assets" / "style.css").write_text(STYLE.strip() + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(render_index(records, output_dir), encoding="utf-8")
    for record in records:
        page_path(output_dir, record["slug"]).write_text(
            render_model_page(record, output_dir), encoding="utf-8"
        )

    print(f"Wrote {len(records)} model page(s) to {output_dir.relative_to(BASE_DIR)}")
    print(f"Open {Path(output_dir / 'index.html').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
