"""Convert the notebooks of a GitHub book repository to Markdown: one folder, one file per notebook.

Run without touching the project environment (standard library only):

    uv run --no-project python scripts/ipynb_to_md.py jakevdp/PythonDataScienceHandbook notebooks \
        --out books-md/vanderplas-2016-python-data-science-handbook

Markdown cells are kept as they are, code cells become fenced `python` blocks, text outputs become `text`
blocks (cut at MAX_OUTPUT_LINES lines) and image outputs are saved into `images/`.
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

USER_AGENT = 'Mozilla/5.0 (study-plan ipynb_to_md)'
MAX_OUTPUT_LINES = 40
PAUSE = 0.3


def get(url: str) -> bytes:
    """Return the body of a URL."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def text_of(source: str | list[str]) -> str:
    """Join a notebook source field into one string."""
    return source if isinstance(source, str) else ''.join(source)


def output_blocks(outputs: list[dict], images: Path, stem: str, counter: list[int]) -> list[str]:
    """Render a code cell's outputs: text as fenced blocks, PNG images saved and linked."""
    blocks = []
    for output in outputs:
        data = output.get('data', {})
        if 'image/png' in data:
            counter[0] += 1
            name = f'{stem}-{counter[0]:02d}.png'
            (images / name).write_bytes(base64.b64decode(text_of(data['image/png'])))
            blocks.append(f'![output](images/{name})')
            continue
        text = text_of(output['text']) if 'text' in output else text_of(data.get('text/plain', ''))
        if text.strip():
            lines = text.rstrip().split('\n')
            if len(lines) > MAX_OUTPUT_LINES:
                lines = [*lines[:MAX_OUTPUT_LINES], '... (output cut)']
            blocks.append('```text\n' + '\n'.join(lines) + '\n```')
    return blocks


def notebook_to_markdown(raw: bytes, images: Path, stem: str) -> str:
    """Convert one notebook's JSON to Markdown."""
    parts: list[str] = []
    counter = [0]
    for cell in json.loads(raw)['cells']:
        source = text_of(cell['source']).rstrip()
        if cell['cell_type'] == 'markdown':
            parts.append(source)
        elif cell['cell_type'] == 'code' and source:
            parts.append('```python\n' + source + '\n```')
            parts.extend(output_blocks(cell.get('outputs', []), images, stem, counter))
    return '\n\n'.join(parts) + '\n'


def main() -> None:
    """Fetch and convert every notebook under a path of a GitHub repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', help='owner/name')
    parser.add_argument('path', help='folder holding the notebooks')
    parser.add_argument('--out', required=True)
    parser.add_argument('--branch', default='master')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    listing = json.loads(get(f'https://api.github.com/repos/{args.repo}/contents/{args.path}?ref={args.branch}'))
    names = sorted(item['name'] for item in listing if item['name'].endswith('.ipynb'))
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    index = [f'# {out.name}', '', f'Source: https://github.com/{args.repo}/tree/{args.branch}/{args.path}', '']
    for name in names:
        raw = get(f'https://raw.githubusercontent.com/{args.repo}/{args.branch}/{args.path}/{name}')
        stem = re.sub(r'[^a-z0-9.]+', '-', Path(name).stem.lower()).strip('-')
        markdown = notebook_to_markdown(raw, out / 'images', stem)
        (out / f'{stem}.md').write_text(markdown, encoding='utf-8')
        first = next((line for line in markdown.split('\n') if line.startswith('#')), stem)
        index.append(f'- [{first.lstrip("# ").strip() or stem}]({stem}.md)')
        print(name, flush=True)
        time.sleep(PAUSE)
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(f'{len(names)} notebooks -> {out}')


if __name__ == '__main__':
    main()
