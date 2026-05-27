import os
import re
import sys
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent

SITE_URL = "https://matbench-discovery.materialsproject.org"
MODELS_URL = f"{SITE_URL}/models"

MODELS_DIR = BASE_DIR / "model_yamls"
PAPERS_DIR = BASE_DIR / "papers"
OUTPUTS_DIR = BASE_DIR / "outputs"
JSON_OUTPUTS_DIR = OUTPUTS_DIR / "json"
TTL_OUTPUTS_DIR = OUTPUTS_DIR / "ttl"

SEED_SCRIPT = BASE_DIR / "seed_kg_open_router.py"
JSON_TO_TTL_SCRIPT = BASE_DIR / "json_to_ttl.py"


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


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


def process_model(model_url):
    print("\n=== MODEL PAGE ===")
    print(model_url)

    model_slug = slugify(model_url.rstrip("/").split("/")[-1])

    html = get_html(model_url)
    soup = BeautifulSoup(html, "html.parser")

    paper_href = find_model_link(soup, "paper")
    files_href = find_model_link(soup, "files")

    if not paper_href:
        print(f"[SKIP] No paper link found for {model_slug}")
        return

    if not files_href:
        print(f"[SKIP] No files link found for {model_slug}")
        return

    paper_url = resolve_pdf_url(urljoin(model_url, paper_href))
    files_url = urljoin(model_url, files_href)

    yaml_urls = find_yaml_links_from_files_url(files_url)

    if not yaml_urls:
        print(f"[SKIP] No YAML found for {model_slug}")
        return

    yaml_url = yaml_urls[0]

    yaml_path = MODELS_DIR / f"{model_slug}.yml"
    pdf_path = PAPERS_DIR / f"{model_slug}.pdf"

    print(f"Paper: {paper_url}")
    print(f"Files: {files_url}")
    print(f"YAML:  {yaml_url}")

    download_file(yaml_url, yaml_path)
    download_file(paper_url, pdf_path)

    output_json = JSON_OUTPUTS_DIR / f"{model_slug}_model_extraction.json"
    output_ttl = TTL_OUTPUTS_DIR / f"{model_slug}_model_individuals_generated.ttl"

    env = os.environ.copy()
    env["YAML_FILE"] = str(yaml_path)
    env["PDF_FILE"] = str(pdf_path)
    env["PDF_URL"] = paper_url
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


MODELS_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TTL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

model_urls = collect_model_urls(limit=None)

print(f"Found {len(model_urls)} model pages.")

for model_url in model_urls:
    try:
        process_model(model_url)
    except Exception as error:
        print(f"[ERROR] {model_url}")
        print(error)