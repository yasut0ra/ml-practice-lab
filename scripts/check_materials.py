"""教材のローカルリンク・見出し・図解・章の構成をオフラインで確認する。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from run_all import CHAPTERS


def without_code(text: str) -> str:
    return re.sub(r"^```.*?^```[^\n]*", "", text, flags=re.M | re.S)


def heading_anchors(text: str) -> set[str]:
    counts: Counter[str] = Counter()
    anchors = set()
    for title in re.findall(r"^#{1,6}\s+(.+)$", without_code(text), flags=re.M):
        base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        number = counts[base]
        anchors.add(f"{base}-{number}" if number else base)
        counts[base] += 1
    return anchors


def check_document(document: Path) -> list[str]:
    errors = []
    source = without_code(document.read_text(encoding="utf-8"))
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", source):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        destination = document.parent / unquote(link.path) if link.path else document
        if not destination.exists():
            errors.append(f"{document.name}: 見つからないリンク {target}")
        elif link.fragment and destination.suffix == ".md":
            if unquote(link.fragment) not in heading_anchors(destination.read_text(encoding="utf-8")):
                errors.append(f"{document.name}: 見つからない見出し {target}")
    return errors


def check_repository(root: Path) -> list[str]:
    errors = []
    documents = [root / "README.md", root / "cheatsheet.md"]
    for folder in ["docs", "exercises", "datasets", "answers"]:
        documents.extend(sorted((root / folder).rglob("*.md")))
    for document in documents:
        errors.extend(check_document(document))

    for chapter in CHAPTERS:
        number = chapter[:4]
        for expected in [root / "scripts" / chapter, root / "answers" / f"{number}_answer.py"]:
            if not expected.is_file():
                errors.append(f"章のファイルが不足: {expected.relative_to(root)}")
        if len(list((root / "exercises").glob(f"{number}_*.md"))) != 1:
            errors.append(f"{number}: 演習ページを1つ用意してください")

    for picture in sorted((root / "docs" / "images").glob("*.svg")):
        try:
            svg = ET.parse(picture).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            if svg.find("svg:title", namespace) is None or svg.find("svg:desc", namespace) is None:
                errors.append(f"{picture.name}: 図のタイトルと代替説明が必要です")
        except ET.ParseError as error:
            errors.append(f"{picture.name}: SVGの形式が不正です: {error}")
    return errors


def main() -> None:
    errors = check_repository(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"教材チェック: {len(CHAPTERS)}章の構成、ローカルリンク・見出し・SVGが正常です。")


if __name__ == "__main__":
    main()
