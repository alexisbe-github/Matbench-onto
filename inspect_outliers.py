from __future__ import annotations

import json
import statistics
from pathlib import Path

from rdflib import Graph, URIRef


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TTL = BASE_DIR / "outputs" / "pschema" / "matbench_ttl_clean.ttl"

PROPERTIES = {
    "hasParameterNumber": "https://k.loria.fr/ontologies/architectureonto#hasParameterNumber",
    "hasNumberOfSamples": "https://k.loria.fr/ontologies/datasetonto#hasNumberOfSamples",
    "hasHiddenDimension": "https://k.loria.fr/ontologies/architectureonto#hasHiddenDimension",
    "hasLayerCount": "https://k.loria.fr/ontologies/architectureonto#hasLayerCount",
    "value": "https://k.loria.fr/ontologies/trainingonto#value",
}


def local_name(uri: object) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def quartiles(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 4:
        return None, None
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[0], qs[2]


def summarize(values: list[tuple[float, str]]) -> dict:
    numbers = [value for value, _subject in values]
    q1, q3 = quartiles(numbers)
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None
    low = (q1 - 1.5 * iqr) if iqr is not None else None
    high = (q3 + 1.5 * iqr) if iqr is not None else None
    outliers = [
        {"value": value, "subject": subject}
        for value, subject in values
        if low is not None and high is not None and (value < low or value > high)
    ]

    return {
        "count": len(values),
        "min": [{"value": value, "subject": subject} for value, subject in values[:5]],
        "max": [{"value": value, "subject": subject} for value, subject in values[-8:]],
        "q1": q1,
        "q3": q3,
        "iqr_low": low,
        "iqr_high": high,
        "iqr_outliers": outliers,
    }


def main() -> None:
    graph = Graph()
    graph.parse(DEFAULT_TTL.as_uri(), format="turtle")

    report = {}
    for label, uri in PROPERTIES.items():
        values: list[tuple[float, str]] = []
        by_subject: dict[str, set[float]] = {}
        for subject, value in graph.subject_objects(URIRef(uri)):
            try:
                numeric_value = float(value)
                subject_name = local_name(subject)
                values.append((numeric_value, subject_name))
                by_subject.setdefault(subject_name, set()).add(numeric_value)
            except (TypeError, ValueError):
                continue
        values.sort()
        if values:
            report[label] = summarize(values)
            report[label]["conflicting_subject_values"] = {
                subject: sorted(subject_values)
                for subject, subject_values in sorted(by_subject.items())
                if len(subject_values) > 1
            }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
