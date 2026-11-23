"""Replace the placeholders (name, e-mail, GitHub user) in the repository files.

Usage: python scripts/personalize.py --name "Jane Doe" --email "jane@example.org" --github janedoe [--repo my-repo-name]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "docs/methods.md", "docs/reproducibility.md", "reports/scientific_report.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--email", required=True)
    ap.add_argument("--github", required=True); ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        t = p.read_text()
        t = t.replace("Your Name", a.name).replace("you@example.com", a.email).replace("YOUR_GITHUB_USERNAME", a.github)
        if a.repo:
            t = t.replace("spatial-ml-xenium-breast", a.repo)
        p.write_text(t)
        print("updated", rel)


if __name__ == "__main__":
    main()
