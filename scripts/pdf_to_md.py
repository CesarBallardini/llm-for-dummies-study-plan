"""Convert book PDFs to Markdown: one folder per book, one file per chapter.

Run without touching the project environment (the converters are not project dependencies):

    uv run --no-project --with pymupdf4llm python scripts/pdf_to_md.py books books/free --out books-md

Arguments may be folders or single PDF files.

Chapters come from the PDF outline. The outline level used is the shallowest one with at least
MIN_CHAPTERS entries. A PDF without an outline is split where a page opens with a heading set much
larger than the body text (the most common such size), then where a "Chapter N" heading starts a page,
or else into fixed blocks of FALLBACK_PAGES pages, and the manifest flags it for manual review.
"""

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MIN_CHAPTERS = 5
FALLBACK_PAGES = 30
PLAIN_HEADING_RE = re.compile(r"[\w ,:'’&.?-]+")
MAX_HEADING_WORDS = 12
MIN_PAGES_PER_CHAPTER = 2  # outline levels whose median entry is shorter are not chapters
SHORT_DOC_PAGES = 100  # below this many pages, fewer headings are enough to split
MIN_SECTIONS_SHORT = 3
HEADING_ZONE = 300  # points from the top of the page where a chapter heading is looked for
HEADING_RATIO = 1.8  # a heading is at least this many times the body font size
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
    if sorted(e - s + 1 for _, s, e in ranges)[len(ranges) // 2] < MIN_PAGES_PER_CHAPTER:
        return fallback_ranges(doc)  # a bookmark per page is a scanner artefact, not chapters
    if len(ranges) < 2 * MIN_CHAPTERS and len(headings := heading_ranges(doc)) > len(ranges):
        return headings, 'large page headings (the outline is incomplete)'
    if ranges and ranges[0][1] > 0:
        ranges.insert(0, ('Front matter', 0, ranges[0][1] - 1))
    return ranges, f'outline level {level}'


def page_headings(doc) -> list[tuple[int, float, str, str]]:  # noqa: ANN001
    """Return (page, size, heading, subtitle) for pages whose top region holds text much larger than the body."""
    pages = []
    sizes: dict[float, int] = {}
    for i in range(len(doc)):
        spans = [
            (round(s['size'], 1), s['text'].strip())
            for b in doc[i].get_text('dict')['blocks']
            if b['type'] == 0 and b['bbox'][1] < HEADING_ZONE
            for ln in b['lines']
            for s in ln['spans']
            if s['text'].strip()
        ]
        pages.append(spans)
        for size, text in spans:
            sizes[size] = sizes.get(size, 0) + len(text)
    body = max(sizes, key=lambda k: sizes[k]) if sizes else 10.0
    found = []
    for i, spans in enumerate(pages):
        big = [s for s in spans if s[0] >= max(HEADING_RATIO * body, body + 6)]
        if not big:
            continue
        size = max(s[0] for s in big)
        heading = ' '.join(t for sz, t in big if sz == size)
        below = [(sz, t) for sz, t in spans if body + 3 <= sz < size]
        subtitle = ' '.join(t for sz, t in below if sz == below[0][0]) if below else ''
        if PLAIN_HEADING_RE.fullmatch(heading) and len(heading.split()) <= MAX_HEADING_WORDS:  # not OCR noise
            found.append((i, size, heading, subtitle))
    return found


def heading_ranges(doc) -> list[tuple[str, int, int]]:  # noqa: ANN001
    """Return chapter ranges from the most common large heading size, or [] when there are too few."""
    found = page_headings(doc)
    if not found:
        return []
    counts: dict[float, int] = {}
    for _, size, _, _ in found:
        counts[size] = counts.get(size, 0) + 1
    mode = max(counts, key=lambda k: counts[k])
    starts = [
        (i, f'Chapter {head} {sub}'.strip() if head.isdigit() else head)
        for i, size, head, sub in found
        if size == mode and re.search(r'\w{2,}|^\d+$', head)
    ]
    if len(starts) < (MIN_CHAPTERS if len(doc) >= SHORT_DOC_PAGES else MIN_SECTIONS_SHORT):
        return []
    ranges = [(title, s, e - 1) for (s, title), e in zip(starts, [*(p for p, _ in starts[1:]), len(doc)], strict=True)]
    if starts[0][0] > 0:
        ranges.insert(0, ('Front matter', 0, starts[0][0] - 1))
    return ranges


def fallback_ranges(doc) -> tuple[list[tuple[str, int, int]], str]:  # noqa: ANN001
    """Split by large page headings, then by 'Chapter N' page headings, else into fixed blocks."""
    ranges = heading_ranges(doc)
    if ranges:
        return ranges, 'large page headings (no outline)'
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


def relocate_images(md: str, scratch: Path, img_dir: Path, pdf_name: str) -> str:
    """Move extracted images from the scratch folder into img_dir with short names and fix the links."""
    for image in list(scratch.iterdir()):
        short = image.name.removeprefix(pdf_name).lstrip('-_.')
        image.replace(img_dir / short)
        for prefix in (scratch.as_posix(), str(scratch)):
            for sep in ('/', '\\'):
                md = md.replace(f'{prefix}{sep}{image.name}', f'images/{short}')
    return md


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
    img_dir.mkdir(exist_ok=True)
    # pymupdf4llm names each image after the PDF, so a long title overruns Windows' 260-character path limit.
    # Images are extracted to a short scratch folder and moved to img_dir under short names.
    scratch = Path(tempfile.mkdtemp(prefix='pmd'))
    index = [f'# {name}', '', f'Source: `{pdf}` ({len(doc)} pages). Chapters found by: {method}.', '']
    for n, (title, first, last) in enumerate(ranges):
        if last < first:
            last = first
        md = pymupdf4llm.to_markdown(
            doc,
            pages=list(range(first, last + 1)),
            write_images=True,
            image_path=str(scratch),
            image_format='jpg',
            dpi=120,
            show_progress=False,
        )
        md = relocate_images(md, scratch, img_dir, Path(pdf).name)
        fname = f'{n:02d}-{slugify(title)}.md'
        (out / fname).write_text(f'# {title}\n\n{md}', encoding='utf-8')
        index.append(f'- [{title}]({fname}) (pages {first + 1}-{last + 1})')
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    shutil.rmtree(scratch, ignore_errors=True)
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
