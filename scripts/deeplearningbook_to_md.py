"""Convert the web edition of Goodfellow, Bengio and Courville's Deep Learning to Markdown.

The site (https://www.deeplearningbook.org) is a pdf2htmlEX render of the printed book: every line of
text is a positioned `div.t`, math symbols are private-use glyphs and the figures are embedded PNGs. The
text is recovered line by line (sub- and superscripts stay attached to their line) and grouped into
paragraphs by a line-length heuristic. Equations are therefore lossy: expect missing symbols and
fraction layouts flattened to text. Use the PDF edition of the book for formulas.

Run without touching the project environment:

    uv run --no-project --with beautifulsoup4 python scripts/deeplearningbook_to_md.py \
        --out books-md/goodfellow-2016-deep-learning
"""

import argparse
import base64
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

BASE = 'https://www.deeplearningbook.org/'
USER_AGENT = 'Mozilla/5.0 (study-plan deeplearningbook_to_md)'
MIN_FIGURE_HEIGHT = 40  # pixels; thinner images are rules and fraction bars
SHORT_LINE = 0.75  # a line shorter than this share of the longest ends its paragraph
PAUSE = 0.5


def fetch(url: str) -> bytes:
    """Return the body of a URL."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read()


def slugify(text: str) -> str:
    """Return a short filesystem-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:60] or 'untitled'


def png_height(data: bytes) -> int:
    """Return the height in pixels of a PNG image."""
    return struct.unpack('>I', data[20:24])[0]


def page_lines(page) -> list[str]:  # noqa: ANN001
    """Return the text lines of one page; a div starting at the left margin (`x0`) starts a new line."""
    lines: list[str] = []
    for div in page.select('div.t'):
        text = div.get_text()
        if 'x0' in div.get('class', []) or not lines:
            lines.append(text)
        else:
            lines[-1] += text
    return [line.strip() for line in lines if line.strip()]


def to_paragraphs(lines: list[str]) -> list[str]:
    """Join text lines into paragraphs: a short line ending in a full stop closes its paragraph."""
    longest = max((len(line) for line in lines), default=0)
    paragraphs: list[str] = []
    current = ''
    for line in lines:
        joined = line if not current else (current[:-1] + line if current.endswith('-') else f'{current} {line}')
        current = joined
        if len(line) < SHORT_LINE * longest and re.search(r'[.:?!]$', line):
            paragraphs.append(current)
            current = ''
    if current:
        paragraphs.append(current)
    return paragraphs


def convert_chapter(html: bytes, title: str, images: Path, stem: str) -> str:
    """Convert one chapter page to Markdown, saving its figures."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

    soup = BeautifulSoup(html, 'html.parser')
    out = [f'# {title}', '']
    for number, page in enumerate(soup.select('div.pf'), 1):
        lines = [
            line
            for line in page_lines(page)
            if not re.fullmatch(r'\d+', line) and not re.match(r'CHAPTER \d+\.', line) and not line.isupper()
        ]
        out.extend(f'{paragraph}\n' for paragraph in to_paragraphs(lines))
        for index, image in enumerate(page.select('img.bi')):
            src = str(image.get('src', ''))
            if not src.startswith('data:image/png;base64,'):
                continue
            data = base64.b64decode(src.split(',', 1)[1])
            if png_height(data) >= MIN_FIGURE_HEIGHT:
                name = f'{stem}-p{number:02d}-{index}.png'
                (images / name).write_bytes(data)
                out.append(f'![page {number} graphics](images/{name})\n')
    return '\n'.join(out)


def main() -> None:
    """Convert every chapter listed on the book's home page."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    home = BeautifulSoup(fetch(BASE), 'html.parser')
    chapters = [
        (a.get_text(' ', strip=True), str(a['href']))
        for a in home.select('a[href^="contents/"]')
        if 'part_' not in str(a['href']) and not str(a['href']).endswith('TOC.html')
    ]
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    index = [f'# {out.name}', '', f'Source: {BASE} (web edition, pdf2htmlEX render; equations are lossy).', '']
    for number, (title, href) in enumerate(chapters):
        stem = f'{number:02d}-{slugify(title)}'
        (out / f'{stem}.md').write_text(
            convert_chapter(fetch(BASE + href), title, out / 'images', stem) + '\n', encoding='utf-8'
        )
        index.append(f'- [{title}]({stem}.md)')
        print(title, flush=True)
        time.sleep(PAUSE)
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
