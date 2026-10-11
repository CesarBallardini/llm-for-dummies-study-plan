"""Convert book PDFs to Markdown: one folder per book, one file per chapter.

Run without touching the project environment (the converters are not project dependencies):

    uv run --no-project --with pymupdf4llm python scripts/pdf_to_md.py books books/free --out books-md

Arguments may be folders or single PDF files.

Chapters come from the PDF outline. The outline level used is the shallowest one with at least
MIN_CHAPTERS entries. A PDF without an outline is split where a "Chapter N" heading starts a page,
or else into fixed blocks of FALLBACK_PAGES pages, and the manifest flags it for manual review.
"""

import argparse
import hashlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MIN_CHAPTERS = 5
FALLBACK_PAGES = 30
CHAPTER_RE = re.compile(r'^\s*(?:chapter|CHAPTER)\s+\d+', re.MULTILINE)


def slugify(text: str) -> str:
    """Return a short filesystem-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug[:60].rstrip('-') or 'untitled'


def chapter_ranges(doc) -> tuple[list[tuple[str, int, int]], str]:  # noqa: ANN001
    """Return [(title, first_page, last_page)] (0-based, inclusive) and how they were found."""
    toc = doc.get_toc()
    levels = sorted({entry[0] for entry in toc})
    level = next((lv for lv in levels if sum(1 for e in toc if e[0] == lv) >= MIN_CHAPTERS), None)
    if level is None:
        return fallback_ranges(doc)
    bounds = [(e[1], e[2] - 1, e[0]) for e in toc if e[0] <= level and e[2] >= 1]
    ranges = []
    for i, (title, start, lv) in enumerate(bounds):
        if lv != level:
            continue
        nxt = next((b[1] for b in bounds[i + 1 :] if b[1] > start), len(doc))
        ranges.append((title.strip(), start, nxt - 1))
    if ranges and ranges[0][1] > 0:
        ranges.insert(0, ('Front matter', 0, ranges[0][1] - 1))
    return ranges, f'outline level {level}'


def fallback_ranges(doc) -> tuple[list[tuple[str, int, int]], str]:  # noqa: ANN001
    """Split by 'Chapter N' page headings, else into fixed blocks."""
    starts = [i for i in range(len(doc)) if CHAPTER_RE.match(doc[i].get_text()[:200] or '')]
    if len(starts) >= MIN_CHAPTERS:
        ends = [*starts[1:], len(doc)]
        ranges = [(f'Chapter {n}', s, e - 1) for n, (s, e) in enumerate(zip(starts, ends, strict=True), 1)]
        if starts[0] > 0:
            ranges.insert(0, ('Front matter', 0, starts[0] - 1))
        return ranges, 'chapter headings (no outline)'
    blocks = [
        (f'Pages {s + 1}-{min(s + FALLBACK_PAGES, len(doc))}', s, min(s + FALLBACK_PAGES, len(doc)) - 1)
        for s in range(0, len(doc), FALLBACK_PAGES)
    ]
    return blocks, f'fixed {FALLBACK_PAGES}-page blocks (no outline) -- REVIEW'


def convert(args: tuple[str, str]) -> tuple[str, str, int, str]:
    """Convert one PDF; return (book, status, chapter_count, method)."""
    import pymupdf  # pyright: ignore[reportMissingImports]
    import pymupdf4llm  # pyright: ignore[reportMissingImports]

    pdf, out_root = args
    name = Path(pdf).stem
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    has_text = any(len(doc[i].get_text()) > 50 for i in range(min(len(doc), 30)))
    ranges, method = chapter_ranges(doc)
    if not has_text:
        method += ' -- NO TEXT LAYER, OCR NEEDED'
    img_dir = out / 'images'
    index = [f'# {name}', '', f'Source: `{pdf}` ({len(doc)} pages). Chapters found by: {method}.', '']
    for n, (title, first, last) in enumerate(ranges):
        if last < first:
            last = first
        md = pymupdf4llm.to_markdown(
            doc,
            pages=list(range(first, last + 1)),
            write_images=True,
            image_path=str(img_dir),
            image_format='jpg',
            dpi=120,
            show_progress=False,
        )
        md = md.replace(str(img_dir).replace('\\', '/'), 'images').replace(str(img_dir), 'images')
        fname = f'{n:02d}-{slugify(title)}.md'
        (out / fname).write_text(f'# {title}\n\n{md}', encoding='utf-8')
        index.append(f'- [{title}]({fname}) (pages {first + 1}-{last + 1})')
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return name, 'ok' if has_text else 'no-text', len(ranges), method


def main() -> None:
    """Convert every distinct PDF found in the given folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--out', default='books-md')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    seen: dict[str, str] = {}
    for folder in args.folders:
        path = Path(folder)
        for pdf in [path] if path.is_file() else sorted(path.glob('*.pdf')):
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            seen.setdefault(digest, str(pdf))
    jobs = [(pdf, args.out) for pdf in seen.values()]
    Path(args.out).mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for name, status, count, method in pool.map(convert, jobs):
            rows.append(f'| `{name}` | {status} | {count} | {method} |')
            print(name, status, count, method, flush=True)
    header = ['# Converted books', '', '| Book | Status | Files | Chapters found by |', '| --- | --- | --- | --- |']
    (Path(args.out) / 'README.md').write_text('\n'.join([*header, *sorted(rows)]) + '\n', encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
