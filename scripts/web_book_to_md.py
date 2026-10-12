r"""Convert multi-page web books to Markdown: one folder per book, one file per page.

Run without touching the project environment (the converters are not project dependencies):

    uv run --no-project --with beautifulsoup4 --with markdownify \
        python scripts/web_book_to_md.py nielsen --out books-md

Sources: nielsen, scaling, smol, lamar, pdsh. Pages are fetched one at a time with a short pause.
Math written as TeX (`$..$`, `\\(..\\)`, `\\[..\\]`, KaTeX annotations, `script type=math/tex`) is kept as `$..$` and
`$$..$$`. Images are downloaded into `images/`. Run with `--limit N` to try the first N pages.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from web_to_md import fetch, slugify

PAUSE = 0.4  # seconds between requests
INLINE_RE = re.compile(r'\\\((.+?)\\\)', re.DOTALL)
DISPLAY_RE = re.compile(r'\\\[(.+?)\\\]|\$\$(.+?)\$\$', re.DOTALL)
DOLLAR_RE = re.compile(r'(?<![\\$])\$([^$\n]{1,300}?)\$(?!\$)')


@dataclass
class Page:
    """One page of a book: its title, address and the folder (chapter) it belongs to."""

    title: str
    url: str
    group: str = ''


@dataclass
class Source:
    """How to read one book: where its pages are, which element holds the text, how its math is written."""

    folder: str
    index_url: str
    selector: str
    dollar_math: bool = False
    drop: tuple[str, ...] = ()
    split_h2: bool = False


def protect_math(soup, root, dollar_math: bool) -> dict[str, str]:  # noqa: ANN001
    """Replace every math expression under root by a token; return the token -> Markdown map."""
    tokens: dict[str, str] = {}

    def keep(text: str) -> str:
        token = f'ZZMATH{len(tokens)}ZZ'
        tokens[token] = text
        return token

    for script in root.select('script[type^="math/tex"]'):
        tex = ' '.join(script.get_text().split())
        script.replace_with(keep(f'\n\n$$ {tex} $$\n\n' if 'display' in str(script.get('type')) else f'${tex}$'))
    for math in root.find_all('d-math'):
        tex = ' '.join(math.get_text().split())
        math.replace_with(keep(f'\n\n$$ {tex} $$\n\n' if math.has_attr('block') else f'${tex}$'))
    for katex in root.select('.katex'):
        note = katex.select_one('annotation[encoding="application/x-tex"]')
        if note is None:
            continue
        tex = ' '.join(note.get_text().split())
        display = katex.find_parent(class_='katex-display') is not None
        target = katex.find_parent(class_='katex-display') or katex
        target.replace_with(keep(f'\n\n$$ {tex} $$\n\n' if display else f'${tex}$'))
    for node in list(root.find_all(string=True)):
        if node.parent is None or node.parent.name in ('script', 'style', 'pre', 'code'):
            continue
        text = str(node)
        if '$' not in text and '\\' not in text:
            continue
        text = DISPLAY_RE.sub(lambda m: keep(f'\n\n$$ {" ".join((m.group(1) or m.group(2)).split())} $$\n\n'), text)
        text = INLINE_RE.sub(lambda m: keep(f'${" ".join(m.group(1).split())}$'), text)
        if dollar_math:
            text = DOLLAR_RE.sub(lambda m: keep(f'${m.group(1).strip()}$'), text)
        if text != str(node):
            node.replace_with(text)
    return tokens


def download_images(root, url: str, images_dir: Path) -> None:  # noqa: ANN001
    """Download every image under root and point its `src` at the local copy."""
    for image in root.find_all('img'):
        src = str(image.get('src', ''))
        if not src or src.startswith('data:'):
            continue
        absolute = urljoin(url, src)
        stem = Path(urlparse(absolute).path)
        name = f'{stem.parent.name}-{stem.name}' if stem.parent.name else stem.name
        target = images_dir / name
        if not target.exists():
            try:
                target.write_bytes(fetch(absolute))
                time.sleep(PAUSE / 4)
            except OSError as error:
                print(f'could not fetch {absolute}: {error}', file=sys.stderr)
                continue
        image['src'] = f'images/{name}'


def page_to_markdown(source: Source, page: Page, out: Path, depth: int) -> str:
    """Fetch one page and return its Markdown; images go to out/images."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
    from markdownify import markdownify  # pyright: ignore[reportMissingImports]

    soup = BeautifulSoup(fetch(page.url), 'html.parser')
    matches = soup.select(source.selector)
    root = max(matches, key=lambda e: len(e.get_text())) if matches else None  # the page body, not a nav block
    if root is None:
        msg = f'{page.url}: no element matches {source.selector!r}'
        raise SystemExit(msg)
    for junk in root.select(', '.join(['script:not([type^="math/tex"])', 'style', 'd-contents', *source.drop])):
        junk.decompose()
    tokens = protect_math(soup, root, source.dollar_math)
    download_images(root, page.url, out / 'images')
    body = markdownify(str(root), heading_style='ATX')
    for token, text in tokens.items():
        body = body.replace(token, text)
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    body = body.replace('](images/', '](' + '../' * depth + 'images/')
    return f'# {page.title}\n\n{body}'


def split_on_h2(title: str, markdown: str) -> list[tuple[str, str]]:
    """Cut one long page into one part per `## ` heading; text before the first heading is the introduction."""
    chunks: list[list[str]] = [[]]
    in_code = False
    for line in markdown.split('\n'):
        if line.startswith('```'):
            in_code = not in_code
        if line.startswith('## ') and not in_code:
            chunks.append([])
        chunks[-1].append(line)
    parts = []
    for lines in chunks:
        heading = re.match(r'## \[?([^\]\n]+)', lines[0]) if lines else None
        parts.append((heading.group(1).strip() if heading else title, '\n'.join(lines).strip()))
    return [part for part in parts if part[1]]


def links(source: Source, pattern: str, base: str | None = None) -> list[Page]:
    """Return the pages linked from the index whose href matches pattern, in order, without duplicates."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

    soup = BeautifulSoup(fetch(source.index_url), 'html.parser')
    pages: list[Page] = []
    seen: set[str] = set()
    for anchor in soup.select('a[href]'):
        href = str(anchor['href']).split('#')[0]
        title = anchor.get_text(' ', strip=True)
        if re.search(pattern, href) and href not in seen and title:
            seen.add(href)
            pages.append(Page(title, urljoin(base or source.index_url, href)))
    return pages


SOURCES: dict[str, Source] = {
    'nielsen': Source(
        'nielsen-2015-neural-networks-and-deep-learning',
        'http://neuralnetworksanddeeplearning.com/',
        'div.section',
        dollar_math=True,
        drop=('#toc',),
    ),
    'scaling': Source(
        'austin-2025-how-to-scale-your-model',
        'https://jax-ml.github.io/scaling-book/',
        'd-article',
        dollar_math=True,
    ),
    'smol': Source(
        'allal-2025-smol-training-playbook',
        'https://huggingfacetb-smol-training-playbook.hf.space/',
        'main',
        split_h2=True,
    ),
    'lamar': Source(
        'dawkins-2026-pauls-online-math-notes',
        'https://tutorial.math.lamar.edu/',
        'main',
        drop=('.sh-all-chip',),
    ),
}
LAMAR_CLASSES = ('Alg', 'CalcI', 'CalcII', 'CalcIII')


def pages_for(name: str, source: Source) -> list[Page]:
    """List the pages of one book, in reading order."""
    if name == 'nielsen':
        return links(source, r'^(about|exercises_and_problems|chap\d|sai|acknowledgements|faq)\.html$')
    if name == 'scaling':
        base = 'https://jax-ml.github.io/scaling-book/'
        return [
            Page(p.title, p.url)
            for p in links(source, r'^/scaling-book/[a-z-]+$', 'https://jax-ml.github.io')
            if p.title.startswith('Part')
        ] or [Page('Index', base)]
    if name == 'smol':
        return [Page('The Smol Training Playbook', source.index_url)]
    pages = []
    for cls in LAMAR_CLASSES:
        landing = Source(source.folder, f'{source.index_url}Classes/{cls}/{cls}.aspx', source.selector)
        pages.extend(
            Page(p.title.split(' In this')[0][:80], p.url, cls)
            for p in links(landing, rf'^/Classes/{cls}/\w+\.aspx$', 'https://tutorial.math.lamar.edu')
        )
    return pages


def main() -> None:
    """Convert the requested book."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('book', choices=[*SOURCES])
    parser.add_argument('--out', default='books-md')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]  # titles contain non-cp1252 characters
    source = SOURCES[args.book]
    out = Path(args.out) / source.folder
    (out / 'images').mkdir(parents=True, exist_ok=True)
    pages = pages_for(args.book, source)
    if args.limit:
        pages = pages[: args.limit]
    index = [f'# {source.folder}', '', f'Source: {source.index_url} (pages fetched {time.strftime("%Y-%m-%d")}).', '']
    counter = 0
    for number, page in enumerate(pages):
        folder = out / page.group if page.group else out
        folder.mkdir(exist_ok=True)
        markdown = page_to_markdown(source, page, out, 1 if page.group else 0)
        parts = split_on_h2(page.title, markdown) if source.split_h2 else [(page.title, markdown)]
        for title, text in parts:
            name = f'{counter:03d}-{slugify(title)}.md'
            counter += 1
            (folder / name).write_text(text + '\n', encoding='utf-8')
            index.append(f'- [{title}]({page.group + "/" if page.group else ""}{name}) <{page.url}>')
        time.sleep(PAUSE)
        print(f'{number + 1}/{len(pages)} {page.title}', flush=True)
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(json.dumps({'book': source.folder, 'pages': len(pages), 'images': len(list((out / 'images').iterdir()))}))


if __name__ == '__main__':
    main()
