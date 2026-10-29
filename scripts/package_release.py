"""Check STaR-DRO release files and optionally build a reproducible review ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".venv", "outputs", "build", "dist", "private", ".ruff_cache", ".pytest_cache"}
ROOT_FILES = {"README.md", "AGENTS.md", "LICENSE", "CITATION.bib", "CITATION.cff", "CONTRIBUTING.md",
              "MANIFEST.in", ".gitignore", "pyproject.toml", "requirements.txt", "requirements-dev.txt", "environment.yml"}
SOURCE_TYPES = {"src": {".py"}, "scripts": {".py"}, "tests": {".py"}, "configs": {".json"},
                "schemas": {".json"}, "prompts": {".txt", ".md"}, ".github": {".yml", ".yaml", ".md"}}
DOCS = {"quickstart.md", "data_format.md", "usage.md", "configuration.md", "method.md", "evaluation.md",
        "reproducibility.md", "validation.json", "data_availability.md", "grouping.md", "results.md",
        "paper_alignment.md", "repository_design.md", "tested-environment.txt"}
SENSITIVE_TYPES = {".arrow", ".parquet", ".jsonl", ".ndjson", ".ipynb", ".pt", ".pth", ".bin", ".safetensors",
                   ".pem", ".key", ".pkl", ".pickle", ".csv", ".tsv"}
REQUIRED = ["README.md", "LICENSE", "CITATION.bib", "CITATION.cff", "pyproject.toml", "requirements.txt",
            "requirements-dev.txt", "environment.yml", "docs/data_format.md", "docs/data_availability.md",
            "docs/method.md", "docs/usage.md", "docs/evaluation.md", "docs/reproducibility.md", "docs/validation.json",
            "configs/reference.json", "AGENTS.md", "docs/quickstart.md", "scripts/run_experiment.py",
            "schemas/record.schema.json", "docs/grouping.md", "docs/paper_alignment.md", "docs/results.md",
            "prompts/structured_prediction.md", "prompts/eppc_miner.md", "src/star_dro/grouping.py",
            "src/star_dro/prompting.py", "scripts/build_records.py"]


def release_files(root=ROOT):
    root = Path(root)
    result = []
    for parent, directories, names in os.walk(root):
        directories[:] = sorted(d for d in directories if d not in EXCLUDED and not d.startswith(".test-") and not d.endswith(".egg-info"))
        for name in sorted(names):
            path = Path(parent)/name
            relative = path.relative_to(root)
            if path.suffix.lower() in SENSITIVE_TYPES or name == ".env" or name.startswith(".env."):
                raise ValueError(f"private data/model/credential file outside excluded directories: {relative}")
            if path.suffix in {".pyc", ".pyo", ".zip"}:
                continue
            if path.is_symlink():
                raise ValueError("symlinks are not included in review releases")
            if ((len(relative.parts) == 1 and name in ROOT_FILES)
                    or (len(relative.parts) == 2 and relative.parts[0] == "docs" and name in DOCS)
                    or (relative.parts[0] in SOURCE_TYPES and path.suffix in SOURCE_TYPES[relative.parts[0]])):
                result.append(path)
    return sorted(result)


def check(files):
    names = {p.relative_to(ROOT).as_posix() for p in files}
    missing = set(REQUIRED)-names
    if missing:
        raise ValueError(f"missing required release files: {sorted(missing)}")
    prohibited = [re.compile(r"hf_"+r"[A-Za-z0-9]{20,}"), re.compile(r"-----BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY-----"),
                  re.compile(r"/nfs/" + r"(?:roberts|[^/]+/scratch)/"),
                  re.compile(r"C:[\\/]Users[\\/]" + r"gp528[\\/]", re.I)]
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if any(pattern.search(text) for pattern in prohibited):
            raise ValueError(f"excluded development reference or credential in {path.relative_to(ROOT)}")
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if not target.startswith(("http://", "https://", "mailto:", "#")):
                    if not (path.parent/target.split("#")[0]).exists():
                        raise ValueError(f"broken link in {path.relative_to(ROOT)}: {target}")
    try:
        import tomllib
    except ImportError:
        tomllib = None
    if tomllib:
        tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf8"))
    return {"files": len(files), "documentation_links": "passed", "json_syntax": "passed",
            "release_content_scan": "passed", "toml_syntax": "passed" if tomllib else "not_checked_python_before_3_11"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new ZIP filename; omit to check files only")
    args = parser.parse_args()
    files = release_files()
    report = check(files)
    if args.output:
        output = Path(args.output)
        if output.exists():
            parser.error("output ZIP exists; choose a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"release": "STaR-DRO", "version": "0.5.0", "checks": report,
                    "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            contents = {"STaR-DRO/"+p.relative_to(ROOT).as_posix(): p.read_bytes() for p in files}
            contents["STaR-DRO/release-manifest.json"] = (json.dumps(manifest,indent=2)+"\n").encode()
            for name, content in sorted(contents.items()):
                info = zipfile.ZipInfo(name, (1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info,content)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        Path(str(output)+".sha256").write_text(digest+"  "+output.name+"\n",encoding="utf8")
        report["zip"] = str(output)
        report["sha256"] = digest
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
