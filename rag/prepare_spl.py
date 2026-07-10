#!/usr/bin/env python3
"""Convert FDA SPL ZIP archives into section-aware JSONL chunks.

The source directory is expected to contain category folders such as
``prescription/`` and ``otc/``. Archives are read in place; images are not
extracted.

Example:
    python -m rag.prepare_spl \
        /path/to/dm_spl_monthly_update_may2026 \
        --category prescription \
        --output data/rag/spl_chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

HL7_NAMESPACE = "urn:hl7-org:v3"
NS = {"hl7": HL7_NAMESPACE}
TAG = f"{{{HL7_NAMESPACE}}}"

DEFAULT_TARGET_TOKENS = 550
DEFAULT_MAX_TOKENS = 850
DEFAULT_OVERLAP_TOKENS = 75
MAX_XML_BYTES = 25 * 1024 * 1024

TOKEN_PATTERN = re.compile(r"\w+(?:[-./]\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
WHITESPACE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES = re.compile(r"\n{3,}")

NOISY_SECTION_PATTERNS = (
    "principal display panel",
    "package label",
    "package/label",
    "questions or comments",
    "questions?",
    "spl product data elements",
)


class SplParseError(Exception):
    """Raised when an SPL archive cannot be converted safely."""


@dataclass(frozen=True)
class ProductMetadata:
    set_id: str
    document_id: str
    version: int | None
    effective_date: str
    document_type: str
    document_type_code: str
    product_name: str
    generic_name: str
    manufacturer: str
    dosage_form: str
    route: str
    ndcs: list[str]
    active_ingredients: list[dict[str, str]]
    source_category: str
    source_archive: str
    source_xml: str


@dataclass(frozen=True)
class SectionDocument:
    section_id: str
    section_code: str
    section_display_name: str
    section_title: str
    parent_section_title: str
    section_effective_date: str
    blocks: list[str]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    token_count_estimate: int
    content_sha256: str
    chunk_index: int
    chunk_count: int
    metadata: dict


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_inline_text(text: str) -> str:
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Return a deterministic tokenizer-independent token estimate."""
    return len(TOKEN_PATTERN.findall(text))


def format_date(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if len(value) >= 8 and value[:8].isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return clean_inline_text("".join(element.itertext()))


def attr(element: ET.Element | None, name: str) -> str:
    return element.get(name, "").strip() if element is not None else ""


def first(root: ET.Element, path: str) -> ET.Element | None:
    return root.find(path, NS)


def all_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def extract_product_metadata(
    root: ET.Element,
    *,
    category: str,
    archive_path: Path,
    xml_name: str,
) -> ProductMetadata:
    document_code = first(root, "hl7:code")
    represented_org = first(
        root,
        "hl7:author/hl7:assignedEntity/hl7:representedOrganization",
    )

    products = root.findall(
        ".//hl7:manufacturedProduct/hl7:manufacturedProduct", NS
    )
    product = products[0] if products else None

    product_name = (
        element_text(first(product, "hl7:name"))
        if product is not None
        else ""
    )
    generic_name = (
        element_text(first(product, "hl7:asEntityWithGeneric/hl7:genericMedicine/hl7:name"))
        if product is not None
        else ""
    )
    dosage_form = (
        attr(first(product, "hl7:formCode"), "displayName")
        if product is not None
        else ""
    )
    route = ""
    if product is not None:
        route = attr(
            first(
                product,
                "hl7:consumedIn/hl7:substanceAdministration/hl7:routeCode",
            ),
            "displayName",
        )

    ndcs: list[str] = []
    active_ingredients: list[dict[str, str]] = []
    if product is not None:
        product_code = first(product, "hl7:code")
        ndcs.append(attr(product_code, "code"))
        for package_code in product.findall(
            ".//hl7:containerPackagedProduct/hl7:code", NS
        ):
            ndcs.append(attr(package_code, "code"))

        for ingredient in product.findall("hl7:ingredient", NS):
            if attr(ingredient, "classCode") != "ACTIB":
                continue
            substance = first(ingredient, "hl7:ingredientSubstance")
            name = (
                element_text(first(substance, "hl7:name"))
                if substance is not None
                else ""
            )
            numerator = first(ingredient, "hl7:quantity/hl7:numerator")
            denominator = first(ingredient, "hl7:quantity/hl7:denominator")
            strength = format_quantity(numerator, denominator)
            active_ingredients.append({"name": name, "strength": strength})

    version_raw = attr(first(root, "hl7:versionNumber"), "value")
    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = None

    return ProductMetadata(
        set_id=attr(first(root, "hl7:setId"), "root"),
        document_id=attr(first(root, "hl7:id"), "root"),
        version=version,
        effective_date=format_date(attr(first(root, "hl7:effectiveTime"), "value")),
        document_type=attr(document_code, "displayName"),
        document_type_code=attr(document_code, "code"),
        product_name=product_name or element_text(first(root, "hl7:title")),
        generic_name=generic_name,
        manufacturer=(
            element_text(first(represented_org, "hl7:name"))
            if represented_org is not None
            else ""
        ),
        dosage_form=dosage_form,
        route=route,
        ndcs=all_unique(ndcs),
        active_ingredients=active_ingredients,
        source_category=category,
        source_archive=str(archive_path),
        source_xml=xml_name,
    )


def format_quantity(
    numerator: ET.Element | None,
    denominator: ET.Element | None,
) -> str:
    if numerator is None:
        return ""
    num_value = attr(numerator, "value")
    num_unit = attr(numerator, "unit")
    den_value = attr(denominator, "value")
    den_unit = attr(denominator, "unit")

    numerator_text = " ".join(part for part in (num_value, num_unit) if part)
    denominator_text = " ".join(part for part in (den_value, den_unit) if part)
    if denominator_text and denominator_text not in {"1", "1 1"}:
        return f"{numerator_text} per {denominator_text}"
    return numerator_text


def render_inline(element: ET.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)

    for child in element:
        name = local_name(child.tag)
        if name == "renderMultiMedia":
            pass
        elif name == "br":
            parts.append("\n")
        elif name == "sup":
            value = clean_inline_text("".join(child.itertext()))
            if value:
                parts.append(f"^{value}")
        elif name == "sub":
            value = clean_inline_text("".join(child.itertext()))
            if value:
                parts.append(f"_{value}")
        else:
            parts.append(render_inline(child))
        if child.tail:
            parts.append(child.tail)

    rendered = "".join(parts)
    rendered = WHITESPACE.sub(" ", rendered)
    rendered = re.sub(r" *\n *", "\n", rendered)
    return rendered.strip()


def render_list(element: ET.Element, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for item in element.findall("hl7:item", NS):
        prefix = "  " * depth + "- "
        inline_parts: list[str] = []
        nested_lists: list[ET.Element] = []
        if item.text:
            inline_parts.append(item.text)
        for child in item:
            if local_name(child.tag) == "list":
                nested_lists.append(child)
            else:
                inline_parts.append(render_inline(child))
            if child.tail:
                inline_parts.append(child.tail)
        item_text = re.sub(r"\s+", " ", " ".join(inline_parts)).strip()
        item_text = re.sub(r"^[•·]\s*", "", item_text)
        if item_text:
            lines.append(prefix + item_text)
        for nested in nested_lists:
            lines.extend(render_list(nested, depth + 1))
    return lines


def render_table(element: ET.Element) -> str:
    rows: list[list[str]] = []
    for row in element.findall(".//hl7:tr", NS):
        cells = [
            clean_inline_text(" ".join(cell.itertext()))
            for cell in list(row)
            if local_name(cell.tag) in {"th", "td"}
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return render_inline(element)

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def render_text_blocks(text_element: ET.Element) -> list[str]:
    blocks: list[str] = []

    def visit(element: ET.Element) -> None:
        name = local_name(element.tag)
        if name == "paragraph":
            text = render_inline(element)
            if text:
                blocks.append(text)
            return
        if name == "list":
            lines = render_list(element)
            if lines:
                blocks.append("\n".join(lines))
            return
        if name == "table":
            table = render_table(element)
            if table:
                blocks.append(table)
            return
        if name in {"renderMultiMedia", "observationMedia"}:
            return
        for child in element:
            visit(child)

    visit(text_element)
    if not blocks:
        fallback = render_inline(text_element)
        if fallback:
            blocks.append(fallback)
    return all_unique(normalize_block(block) for block in blocks)


def normalize_block(block: str) -> str:
    lines = [line.rstrip() for line in block.splitlines()]
    return BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def is_noisy_section(title: str, display_name: str) -> bool:
    normalized = f"{title} {display_name}".lower()
    return any(pattern in normalized for pattern in NOISY_SECTION_PATTERNS)


def infer_subsection_title(blocks: list[str]) -> tuple[str, list[str]]:
    if len(blocks) < 2:
        return "", blocks
    candidate = re.sub(r"\s+", " ", blocks[0]).strip(" :")
    if (
        candidate
        and len(candidate) <= 120
        and estimate_tokens(candidate) <= 16
        and "\n" not in candidate
    ):
        return candidate, blocks[1:]
    return "", blocks


def extract_sections(root: ET.Element) -> list[SectionDocument]:
    sections: list[SectionDocument] = []

    def visit(section: ET.Element, parent_title: str = "") -> None:
        text_element = first(section, "hl7:text")
        code_element = first(section, "hl7:code")
        title = element_text(first(section, "hl7:title"))
        display_name = attr(code_element, "displayName")
        effective_title = title or display_name or "Untitled section"

        if text_element is not None and not is_noisy_section(title, display_name):
            blocks = render_text_blocks(text_element)
            if "unclassified section" in effective_title.lower():
                inferred_title, blocks = infer_subsection_title(blocks)
                if inferred_title:
                    effective_title = inferred_title

            joined = "\n\n".join(blocks)
            boilerplate = joined.lower().lstrip()
            is_boilerplate = (
                estimate_tokens(joined) < 80
                and boilerplate.startswith(
                    ("distributed by", "repackaged by", "manufactured by")
                )
            )
            if blocks and estimate_tokens(joined) >= 8 and not is_boilerplate:
                sections.append(
                    SectionDocument(
                        section_id=attr(first(section, "hl7:id"), "root")
                        or attr(section, "ID"),
                        section_code=attr(code_element, "code"),
                        section_display_name=display_name,
                        section_title=effective_title,
                        parent_section_title=parent_title,
                        section_effective_date=format_date(
                            attr(
                                first(section, "hl7:effectiveTime"),
                                "value",
                            )
                        ),
                        blocks=blocks,
                    )
                )

        for child in section.findall("hl7:component/hl7:section", NS):
            visit(child, effective_title)

    structured_body = first(root, "hl7:component/hl7:structuredBody")
    if structured_body is None:
        return sections
    for section in structured_body.findall("hl7:component/hl7:section", NS):
        visit(section)
    return sections


def split_sentences(text: str) -> list[str]:
    if text.startswith("| ") or "\n| " in text:
        return split_table_rows(text)
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text)]
    return [sentence for sentence in sentences if sentence]


def split_table_rows(table: str) -> list[str]:
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) <= 2:
        return [table]
    header = lines[:2]
    return ["\n".join(header + [row]) for row in lines[2:]]


def tail_for_overlap(parts: Sequence[str], overlap_tokens: int) -> list[str]:
    selected: list[str] = []
    total = 0
    for part in reversed(parts):
        part_tokens = estimate_tokens(part)
        if selected and total + part_tokens > overlap_tokens:
            break
        selected.append(part)
        total += part_tokens
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def split_oversized_part(
    part: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    units = split_sentences(part)

    def split_by_words(value: str) -> list[str]:
        words = value.split()
        groups: list[str] = []
        current: list[str] = []
        for word in words:
            if current and estimate_tokens(" ".join(current + [word])) > max_tokens:
                groups.append(" ".join(current))
                current = tail_for_overlap(current, overlap_tokens)
                while current and estimate_tokens(
                    " ".join(current + [word])
                ) > max_tokens:
                    current.pop(0)
            current.append(word)
        if current:
            groups.append(" ".join(current))
        return groups

    if len(units) == 1 and estimate_tokens(part) > max_tokens:
        return split_by_words(part)

    expanded_units: list[str] = []
    for unit in units:
        if estimate_tokens(unit) > max_tokens:
            expanded_units.extend(split_by_words(unit))
        else:
            expanded_units.append(unit)

    groups: list[str] = []
    current: list[str] = []
    for unit in expanded_units:
        candidate = " ".join(current + [unit])
        if current and estimate_tokens(candidate) > max_tokens:
            groups.append(" ".join(current))
            current = tail_for_overlap(current, overlap_tokens)
            while current and estimate_tokens(
                " ".join(current + [unit])
            ) > max_tokens:
                current.pop(0)
        current.append(unit)
    if current:
        groups.append(" ".join(current))
    return groups


def chunk_blocks(
    blocks: Sequence[str],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    parts: list[str] = []
    for block in blocks:
        if estimate_tokens(block) > max_tokens:
            parts.extend(
                split_oversized_part(
                    block,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        else:
            parts.append(block)

    chunks: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = "\n\n".join(current + [part])
        candidate_tokens = estimate_tokens(candidate)
        should_flush = (
            current
            and estimate_tokens("\n\n".join(current)) >= target_tokens
            and candidate_tokens > target_tokens
        )
        exceeds_max = current and candidate_tokens > max_tokens
        if should_flush or exceeds_max:
            chunks.append("\n\n".join(current))
            current = tail_for_overlap(current, overlap_tokens)
            while current and estimate_tokens(
                "\n\n".join(current + [part])
            ) > max_tokens:
                current.pop(0)
        current.append(part)
    if current:
        chunks.append("\n\n".join(current))
    return all_unique(normalize_block(chunk) for chunk in chunks if chunk.strip())


def build_chunks(
    metadata: ProductMetadata,
    sections: Sequence[SectionDocument],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    output: list[Chunk] = []
    metadata_dict = asdict(metadata)
    for section in sections:
        qualified_section = section.section_title
        if (
            section.parent_section_title
            and section.parent_section_title != section.section_title
        ):
            qualified_section = (
                f"{section.parent_section_title} > {section.section_title}"
            )
        context_header = (
            f"Drug: {metadata.product_name or metadata.generic_name}\n"
            f"Section: {qualified_section}\n\n"
        )
        header_tokens = estimate_tokens(context_header)
        section_chunks = chunk_blocks(
            section.blocks,
            target_tokens=max(50, target_tokens - header_tokens),
            max_tokens=max(100, max_tokens - header_tokens),
            overlap_tokens=overlap_tokens,
        )
        chunk_count = len(section_chunks)
        for index, body in enumerate(section_chunks):
            text = context_header + body
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            identity = "|".join(
                [
                    metadata.set_id,
                    str(metadata.version or ""),
                    section.section_code or section.section_title,
                    str(index),
                    digest,
                ]
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            chunk_metadata = {
                **metadata_dict,
                "section_id": section.section_id,
                "section_code": section.section_code,
                "section_display_name": section.section_display_name,
                "section_title": section.section_title,
                "parent_section_title": section.parent_section_title,
                "qualified_section_title": qualified_section,
                "section_effective_date": section.section_effective_date,
            }
            output.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    token_count_estimate=estimate_tokens(text),
                    content_sha256=digest,
                    chunk_index=index,
                    chunk_count=chunk_count,
                    metadata=chunk_metadata,
                )
            )
    return output


def read_spl_archive(
    archive_path: Path,
    *,
    category: str,
) -> tuple[ProductMetadata, list[SectionDocument]]:
    with zipfile.ZipFile(archive_path) as archive:
        xml_names = [
            name for name in archive.namelist() if name.lower().endswith(".xml")
        ]
        if len(xml_names) != 1:
            raise SplParseError(
                f"Expected exactly one XML file, found {len(xml_names)}"
            )
        xml_name = xml_names[0]
        info = archive.getinfo(xml_name)
        if info.file_size > MAX_XML_BYTES:
            raise SplParseError(
                f"XML exceeds {MAX_XML_BYTES} byte safety limit"
            )
        root = ET.fromstring(archive.read(xml_name))

    metadata = extract_product_metadata(
        root,
        category=category,
        archive_path=archive_path,
        xml_name=xml_name,
    )
    return metadata, extract_sections(root)


def iter_archives(
    source: Path,
    categories: Sequence[str],
    limit: int | None,
) -> Iterator[tuple[str, Path]]:
    count = 0
    for category in categories:
        category_dir = source / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")
        for archive_path in sorted(category_dir.glob("*.zip")):
            yield category, archive_path
            count += 1
            if limit is not None and count >= limit:
                return


def prepare_corpus(
    *,
    source: Path,
    output: Path,
    categories: Sequence[str],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    limit: int | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    errors: list[dict[str, str]] = []

    with output.open("w", encoding="utf-8") as output_file:
        for category, archive_path in iter_archives(source, categories, limit):
            stats["archives_seen"] += 1
            try:
                metadata, sections = read_spl_archive(
                    archive_path, category=category
                )
                chunks = build_chunks(
                    metadata,
                    sections,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
                for chunk in chunks:
                    output_file.write(
                        json.dumps(asdict(chunk), ensure_ascii=False) + "\n"
                    )
                stats["archives_processed"] += 1
                stats["sections"] += len(sections)
                stats["chunks"] += len(chunks)
            except Exception as exc:
                stats["archives_failed"] += 1
                errors.append(
                    {"archive": str(archive_path), "error": str(exc)}
                )

            if stats["archives_seen"] % 100 == 0:
                print(
                    f"Processed {stats['archives_seen']} archives, "
                    f"wrote {stats['chunks']} chunks",
                    file=sys.stderr,
                )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source": str(source),
        "output": str(output),
        "categories": list(categories),
        "chunking": {
            "strategy": "SPL section -> structural blocks -> token estimate",
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "tokenizer": "regex estimate; replace with embedding-model tokenizer for exact counts",
        },
        "stats": dict(stats),
        "errors": errors,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FDA SPL labels as JSONL chunks for RAG."
    )
    parser.add_argument("source", type=Path, help="Monthly SPL update directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rag/spl_chunks.jsonl"),
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Category to process; repeat for multiple categories",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N archives for testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = args.categories or ["prescription"]
    if args.target_tokens <= 0 or args.max_tokens < args.target_tokens:
        raise SystemExit("Require 0 < target_tokens <= max_tokens")
    if not 0 <= args.overlap_tokens < args.target_tokens:
        raise SystemExit("Require 0 <= overlap_tokens < target_tokens")

    manifest = prepare_corpus(
        source=args.source,
        output=args.output,
        categories=categories,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
