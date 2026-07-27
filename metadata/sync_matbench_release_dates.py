import argparse
import re
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OFFICIAL_REPOSITORY = "https://github.com/janosh/matbench-discovery"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/janosh/"
    "matbench-discovery/main"
)


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def read_top_level_field(path: Path, field: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(field)}:\s*(?:['\"]([^'\"]+)['\"]|([^#\s]+))\s*(?:#.*)?$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1) or match.group(2)
    return None


def read_field(path: Path, field: str) -> str | None:
    pattern = re.compile(
        rf"^\s*{re.escape(field)}:\s*(?:['\"]([^'\"]+)['\"]|([^#\s]+))\s*(?:#.*)?$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1) or match.group(2)
    return None


def official_metadata(source_root: Path) -> dict[str, tuple[str, str]]:
    metadata = {}
    for path in sorted((source_root / "models").rglob("*.yml")):
        model_key = read_top_level_field(path, "model_key")
        release_date = read_field(path, "benchmark_added")
        if not model_key or not release_date:
            continue

        # Parsing rejects invalid dates such as 2025-02-30.
        normalized_date = date.fromisoformat(release_date).isoformat()
        relative_path = path.relative_to(source_root).as_posix()
        metadata[canonical_key(model_key)] = (
            normalized_date,
            f"{RAW_BASE_URL}/{relative_path}",
        )
    return metadata


def evaluated_variant(ttl_path: Path) -> str:
    text = ttl_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"eval:evaluatesModelVariant\s+archind:([A-Za-z0-9_-]+)",
        text,
    )
    variants = set(matches)
    if len(variants) != 1:
        raise ValueError(
            f"Expected one evaluated variant in {ttl_path}, found {variants}"
        )
    return variants.pop()


def turtle_document(records: list[tuple[str, str, str]]) -> str:
    lines = [
        "@prefix arch: <https://k.loria.fr/ontologies/architectureonto#> .",
        "@prefix archind: <https://k.loria.fr/ontologies/architectureonto-individuals#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# Dates synchronized from the official Matbench Discovery model metadata.",
        "",
    ]
    for variant, release_date, source_url in sorted(records):
        lines.extend(
            [
                f"archind:{variant}",
                f'    arch:hasReleaseDate "{release_date}"^^xsd:date ;',
                f"    dcterms:source <{source_url}> .",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize Matbench Discovery model release dates into Turtle."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root of an official matbench-discovery checkout.",
    )
    parser.add_argument(
        "--local-yaml-dir",
        type=Path,
        default=BASE_DIR / "model_yamls",
    )
    parser.add_argument(
        "--ttl-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "ttl_repaired",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "outputs" / "ttl_repaired" / "matbench_release_dates.ttl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official = official_metadata(args.source_root)
    records = []
    missing = []
    missing_ttl = []

    for local_yaml in sorted(args.local_yaml_dir.glob("*.yml")):
        model_key = read_top_level_field(local_yaml, "model_key")
        if not model_key:
            raise ValueError(f"Missing model_key in {local_yaml}")
        official_key = canonical_key(model_key)
        if official_key not in official:
            missing.append(model_key)
            continue

        ttl_path = (
            args.ttl_dir
            / f"{local_yaml.stem}_model_individuals_generated.ttl"
        )
        if not ttl_path.exists():
            missing_ttl.append(model_key)
            continue

        release_date, source_url = official[official_key]
        local_date = read_top_level_field(local_yaml, "date_added")
        if local_date and date.fromisoformat(local_date).isoformat() != release_date:
            print(
                f"[UPDATED SOURCE] {model_key}: "
                f"{local_date} -> {release_date}"
            )

        records.append(
            (evaluated_variant(ttl_path), release_date, source_url)
        )

    if missing:
        raise ValueError(
            "Models missing from the official metadata: "
            + ", ".join(missing)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(turtle_document(records), encoding="utf-8")
    print(f"[OK] Wrote {len(records)} release dates to {args.output}")
    if missing_ttl:
        print(
            "[SKIPPED: no local KG individual] "
            + ", ".join(missing_ttl)
        )
    print(f"[SOURCE] {OFFICIAL_REPOSITORY}")


if __name__ == "__main__":
    main()
