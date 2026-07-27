import argparse
import json
import os
import re
import sys
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent.parent

SITE_URL = "https://matbench-discovery.materialsproject.org"
MODELS_URL = f"{SITE_URL}/models"

MODELS_DIR = BASE_DIR / "model_yamls"
PAPERS_DIR = BASE_DIR / "papers"
OUTPUTS_DIR = BASE_DIR / "outputs"
JSON_OUTPUTS_DIR = OUTPUTS_DIR / "json"
TTL_OUTPUTS_DIR = OUTPUTS_DIR / "ttl"
MODEL_PAGE_OUTPUTS_DIR = OUTPUTS_DIR / "model_pages"

SEED_SCRIPT = BASE_DIR / "pipeline" / "seed_kg_open_router.py"
JSON_TO_TTL_SCRIPT = BASE_DIR / "pipeline" / "json_to_ttl.py"

FALLBACK_PDF_BY_MODEL = {
    "grace_1l_oam": "grace_2l_oam_l.pdf",
    "grace_2l_oam": "grace_2l_oam_l.pdf",
    "grace_2l_mptrj": "grace_2l_oam_l.pdf",
}


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def model_slug_from_url(model_url):
    return slugify(model_url.rstrip("/").split("/")[-1])


def normalize_model_selector(value):
    if not value:
        return None

    value = str(value).strip()

    if value.endswith((".yml", ".yaml", ".pdf", ".json", ".ttl")):
        value = Path(value).stem

    for suffix in (
        "_model_extraction",
        "_model_individuals_generated",
        "_individuals_generated",
    ):
        if value.endswith(suffix):
            value = value[: -len(suffix)]

    return model_slug_from_url(value) if "/models/" in value else slugify(value)


def slice_model_urls(model_urls, start_from=None):
    start_slug = normalize_model_selector(start_from)

    if not start_slug:
        return model_urls

    for index, model_url in enumerate(model_urls):
        if model_slug_from_url(model_url) == start_slug:
            return model_urls[index:]

    available = ", ".join(model_slug_from_url(url) for url in model_urls[:10])
    raise ValueError(
        f"Could not find start model '{start_from}' "
        f"(normalized as '{start_slug}'). First available models: {available}"
    )


def get_html(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()

        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


def collect_model_urls(limit=None):
    html = get_html(MODELS_URL)
    soup = BeautifulSoup(html, "html.parser")

    urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if href.startswith("/models/"):
            full_url = urljoin(SITE_URL, href).split("#")[0]
            urls.add(full_url)

    urls = sorted(urls)

    if limit:
        urls = urls[:limit]

    return urls


def find_model_link(soup, kind):
    model_detail = soup.find("div", class_=lambda c: c and "model-detail" in c)

    if not model_detail:
        return None

    links_section = model_detail.find(
        "section",
        class_=lambda c: c and "links" in c
    )

    if not links_section:
        return None

    for link in links_section.find_all("a", href=True):
        text = link.get_text(" ", strip=True).lower()
        title = link.get("data-original-title", "").lower()

        if kind == "paper":
            if text == "paper" or "read model paper" in title:
                return link["href"]

        if kind == "files":
            if text == "files" or "browse model submission files" in title:
                return link["href"]

    return None


def normalize_paper_url(url):
    if not url:
        return None

    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"

    if "arxiv.org/html/" in url:
        arxiv_id = url.rstrip("/").split("/")[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return url


def clean_page_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def extract_model_page_context(soup, model_url):
    root = soup.find("main") or soup.find(
        "div", class_=lambda value: value and "model-detail" in value
    ) or soup.body

    if root is None:
        return {"url": model_url, "sections": []}

    for tag in root.find_all(["script", "style", "noscript"]):
        tag.decompose()

    page_text = clean_page_text(root.get_text(" ", strip=True))
    sections = [{
        "name": "model_page",
        "source": model_url,
        "text": page_text[:30000],
    }]

    steps_node = root.find(
        string=lambda value: value
        and "training performed by" in value.lower()
    )

    if steps_node is not None:
        steps_text = clean_page_text(str(steps_node))

        for parent in steps_node.parents:
            if parent == root:
                break

            candidate = clean_page_text(parent.get_text(" ", strip=True))
            if len(steps_text) < len(candidate) <= 4000:
                steps_text = candidate

            if len(steps_text) >= 120:
                break

        sections.append({
            "name": "steps",
            "source": f"{model_url}#steps",
            "text": steps_text,
        })

    return {
        "url": model_url,
        "sections": sections,
    }


def save_model_page_context(context, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(context, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_pdf_url(paper_url):
    paper_url = normalize_paper_url(paper_url)

    if paper_url.endswith(".pdf"):
        return paper_url

    html = get_html(paper_url)
    soup = BeautifulSoup(html, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if ".pdf" in href or href.endswith(".pdf"):
            return urljoin(paper_url, href)

    return paper_url


def github_directory_to_api_url(github_url):
    parts = github_url.split("github.com/")[-1].split("/")

    owner = parts[0]
    repo = parts[1]

    if "tree" in parts:
        idx = parts.index("tree")
    elif "blob" in parts:
        idx = parts.index("blob")
    else:
        return None

    branch = parts[idx + 1]

    if branch == "-":
        branch = "main"

    path = "/".join(parts[idx + 2:])

    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"


def find_yaml_links_from_files_url(files_url):
    yaml_urls = []

    if files_url.endswith((".yml", ".yaml")):
        return [files_url]

    if "github.com" in files_url:
        api_url = github_directory_to_api_url(files_url)

        if not api_url:
            return []

        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        files = response.json()

        if isinstance(files, dict):
            files = [files]

        for file in files:
            name = file.get("name", "")
            download_url = file.get("download_url")

            if name.endswith((".yml", ".yaml")) and download_url:
                yaml_urls.append(download_url)

        return yaml_urls

    html = get_html(files_url)
    soup = BeautifulSoup(html, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if href.endswith((".yml", ".yaml")):
            yaml_urls.append(urljoin(files_url, href))

    return yaml_urls


def select_yaml_url(yaml_urls, model_slug):
    if not yaml_urls:
        return None

    scored_urls = []

    for index, yaml_url in enumerate(yaml_urls):
        filename = yaml_url.rstrip("/").split("/")[-1]
        stem = filename.rsplit(".", 1)[0]
        yaml_slug = slugify(stem)

        score = 0

        if yaml_slug == model_slug:
            score += 100
        elif model_slug in yaml_slug or yaml_slug in model_slug:
            score += 50

        model_tokens = set(model_slug.split("_"))
        yaml_tokens = set(yaml_slug.split("_"))
        score += len(model_tokens & yaml_tokens)

        scored_urls.append((score, -index, yaml_url))

    scored_urls.sort(reverse=True)
    return scored_urls[0][2]


def process_model(model_url):
    print("\n=== MODEL PAGE ===")
    print(model_url)

    model_slug = model_slug_from_url(model_url)

    html = get_html(model_url)
    soup = BeautifulSoup(html, "html.parser")
    model_page_path = MODEL_PAGE_OUTPUTS_DIR / f"{model_slug}_model_page.json"
    save_model_page_context(
        extract_model_page_context(soup, model_url),
        model_page_path,
    )

    paper_href = find_model_link(soup, "paper")
    files_href = find_model_link(soup, "files")

    if not paper_href:
        print(f"[SKIP] No paper link found for {model_slug}")
        return

    if not files_href:
        print(f"[SKIP] No files link found for {model_slug}")
        return

    files_url = urljoin(model_url, files_href)

    yaml_urls = find_yaml_links_from_files_url(files_url)

    if not yaml_urls:
        print(f"[SKIP] No YAML found for {model_slug}")
        return

    yaml_url = select_yaml_url(yaml_urls, model_slug)

    pdf_path = PAPERS_DIR / f"{model_slug}.pdf"
    yaml_path = MODELS_DIR / f"{model_slug}.yml"

    fallback_pdf_name = FALLBACK_PDF_BY_MODEL.get(model_slug)
    fallback_pdf_path = PAPERS_DIR / fallback_pdf_name if fallback_pdf_name else None

    if pdf_path.exists():
        paper_url = normalize_paper_url(urljoin(model_url, paper_href))
        print(f"[CACHE] Paper already exists: {pdf_path}")
    else:
        try:
            paper_url = resolve_pdf_url(urljoin(model_url, paper_href))
        except Exception as error:
            if fallback_pdf_path and fallback_pdf_path.exists():
                paper_url = normalize_paper_url(urljoin(model_url, paper_href))
                pdf_path = fallback_pdf_path
                print(f"[WARN] Paper download URL failed for {model_slug}: {error}")
                print(f"[FALLBACK] Using local paper: {pdf_path}")
            else:
                raise

    print(f"Paper: {paper_url}")
    print(f"Files: {files_url}")
    print(f"YAML:  {yaml_url}")

    download_file(yaml_url, yaml_path)

    if not pdf_path.exists():
        download_file(paper_url, pdf_path)

    output_json = JSON_OUTPUTS_DIR / f"{model_slug}_model_extraction.json"
    output_ttl = TTL_OUTPUTS_DIR / f"{model_slug}_model_individuals_generated.ttl"

    env = os.environ.copy()
    env["YAML_FILE"] = str(yaml_path)
    env["PDF_FILE"] = str(pdf_path)
    env["PDF_URL"] = paper_url
    env["MODEL_PAGE_FILE"] = str(model_page_path)
    env["MODEL_PAGE_URL"] = model_url
    env["OUTPUT_JSON_FILE"] = str(output_json)

    print("Running LLM extraction...")
    subprocess.run(
        [sys.executable, str(SEED_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        check=True,
    )

    env = os.environ.copy()
    env["INPUT_JSON_FILE"] = str(output_json)
    env["OUTPUT_TTL_FILE"] = str(output_ttl)

    print("Generating TTL...")
    subprocess.run(
        [sys.executable, str(JSON_TO_TTL_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        check=True,
    )

    print(f"[OK] JSON: {output_json}")
    print(f"[OK] TTL:  {output_ttl}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Matbench Discovery models and generate JSON/TTL individuals."
    )
    parser.add_argument(
        "--start-from",
        help=(
            "Resume from this model, inclusive. Accepts a model slug, model URL, "
            "YAML/PDF filename, JSON extraction filename, or TTL filename."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many model pages after applying --start-from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TTL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PAGE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    all_model_urls = collect_model_urls(limit=None)
    model_urls = slice_model_urls(all_model_urls, start_from=args.start_from)

    if args.limit:
        model_urls = model_urls[: args.limit]

    print(f"Found {len(all_model_urls)} model pages.")

    if args.start_from:
        print(
            f"Resuming from {model_slug_from_url(model_urls[0])}: "
            f"{len(model_urls)} model page(s) queued."
        )

    for model_url in model_urls:
        try:
            process_model(model_url)
        except Exception as error:
            print(f"[ERROR] {model_url}")
            print(error)


if __name__ == "__main__":
    main()
