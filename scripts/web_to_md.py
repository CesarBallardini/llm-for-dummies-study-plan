"""Convert a Distill-style web book to Markdown: one folder, one file per top-level section.

Run without touching the project environment (the converters are not project dependencies):

    uv run --no-project --with beautifulsoup4 --with markdownify python scripts/web_to_md.py \
        https://nanotron-ultrascale-playbook.static.hf.space/index.html \
        --out books-md/tazi-2025-ultra-scale-playbook-web

The article body is the `d-article` element. Each `h2` starts a new file; text before the first `h2`
becomes `00-introduction.md`. Math (`d-math`) becomes `$...$` or `$$...$$`, citations (`d-cite`)
become `[key]`, code blocks keep their language, interactive iframes become a link, and images are
downloaded into `images/` and referenced relatively.
"""

import argparse
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

USER_AGENT = 'Mozilla/5.0 (study-plan web_to_md)'


def fetch(url: str) -> bytes:
    """Return the body of a URL."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def slugify(text: str) -> str:
    """Return a short filesystem-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug[:60].rstrip('-') or 'untitled'


def protect_special_elements(soup, article, url: str) -> dict[str, str]:  # noqa: ANN001
    """Replace math, citations, code and iframes by tokens; return the token -> Markdown map.

    The HTML converter escapes `_` and `*` in text, which would corrupt math, so the Markdown for
    these elements is restored after conversion.
    """
    tokens: dict[str, str] = {}

    def keep(text: str) -> str:
        token = f'ZZKEEP{len(tokens)}ZZ'
        tokens[token] = text
        return token

    for math in article.find_all('d-math'):
        tex = ' '.join(math.get_text().split())
        math.replace_with(keep(f'\n\n$$ {tex} $$\n\n' if math.has_attr('block') else f'${tex}$'))
    for cite in article.find_all('d-cite'):
        cite.replace_with(keep(f'[{cite.get("bibtex-key", "cite")}]'))
    for code in article.find_all('d-code'):
        pre = soup.new_tag('pre')
        inner = soup.new_tag('code')
        pre['data-lang'] = str(code.get('language', ''))
        inner.string = code.get_text()
        pre.append(inner)
        code.replace_with(pre)
    for frame in article.find_all('iframe'):
        frame.replace_with(keep(f'\n\n*[Interactive figure: {urljoin(url, str(frame.get("src", "")))}]*\n\n'))
    for tag in article.find_all(['script', 'd-contents', 'style']):
        tag.decompose()
    return tokens


def download_images(article, url: str, images_dir: Path) -> None:  # noqa: ANN001
    """Download every image and point its `src` at the local copy."""
    for image in article.find_all('img'):
        absolute = urljoin(url, str(image.get('src', '')))
        name = Path(urlparse(absolute).path).name or 'image'
        target = images_dir / name
        if not target.exists():
            try:
                target.write_bytes(fetch(absolute))
            except OSError as error:
                print(f'could not fetch {absolute}: {error}')
                continue
        image['src'] = f'images/{name}'


def split_sections(article) -> list[tuple[str, list]]:  # noqa: ANN001
    """Group the article's children into sections that each start at an `h2`."""
    sections: list[tuple[str, list]] = [('Introduction', [])]
    for child in list(article.children):
        if getattr(child, 'name', None) == 'h2':
            sections.append((child.get_text(' ', strip=True), []))
        sections[-1][1].append(child)
    return sections


def main() -> None:
    """Convert the page given on the command line."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
    from markdownify import markdownify  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--out', required=True)
    parser.add_argument('--selector', default='d-article')
    args = parser.parse_args()

    soup = BeautifulSoup(fetch(args.url), 'html.parser')
    article = soup.select_one(args.selector)
    if article is None:
        msg = f'no element matches {args.selector!r}'
        raise SystemExit(msg)
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)

    tokens = protect_special_elements(soup, article, args.url)
    download_images(article, args.url, out / 'images')
    sections = split_sections(article)

    index = [f'# {out.name}', '', f'Source: {args.url}', '']
    for number, (title, nodes) in enumerate(sections):
        body = markdownify(
            ''.join(str(node) for node in nodes),
            heading_style='ATX',
            code_language_callback=lambda el: el.get('data-lang') or None,
        )
        for token, text in tokens.items():
            body = body.replace(token, text)
        body = re.sub(r'\n{3,}', '\n\n', body).strip()
        if not body.lstrip().startswith('#'):
            body = f'# {title}\n\n{body}'
        name = f'{number:02d}-{slugify(title)}.md'
        (out / name).write_text(body + '\n', encoding='utf-8')
        index.append(f'- [{title}]({name})')
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(f'{len(sections)} sections, {len(list((out / "images").iterdir()))} images -> {out}')


if __name__ == '__main__':
    main()
