"""Ask questions about Copperline Manufacturing's finance documents (RAG).

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-03\\ask.py "What is the dinner limit in Chicago?"
    .venv\\Scripts\\python.exe week-03\\ask.py              # interactive: ask several questions
    .venv\\Scripts\\python.exe week-03\\ask.py --rebuild    # re-index after adding/removing documents

How it works:
  1. Every supported file in copperline-docs/ (HTML, PDF, Word, Markdown, text) is split
     into one chunk per section. Each chunk starts with "<document title> - <section heading>"
     so it makes sense on its own, and records the file it came from.
  2. Chunks are embedded with Chroma's built-in embedding model and stored in chroma_db/.
     The index is rebuilt automatically when files in copperline-docs/ change, or on --rebuild.
  3. For each question the 4 most similar chunks are retrieved and sent to Claude, which
     answers only from those sources and cites the source file for each fact.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "copperline-docs"
DB_DIR = HERE / "chroma_db"
COLLECTION = "copperline_docs"
MODEL = "claude-sonnet-5"
TOP_K = 4
PREVIEW_CHARS = 150
MAX_CHUNK_CHARS = 1500   # sections longer than this are split on paragraph boundaries
MIN_INTRO_WORDS = 25     # text before the first heading is kept only if it's this long
CHUNKER_VERSION = "1"    # bump to force a rebuild when the chunking logic changes

# The key file lives in the folder that contains the repository, never inside it.
KEY_FILES = [HERE.parent.parent / "API-Key.txt", HERE.parent.parent / "API Key.txt"]

SUPPORTED = {".html", ".htm", ".pdf", ".docx", ".md", ".markdown", ".txt"}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_site_suffix(title):
    """'Corporate Card FAQ | Copperline Manufacturing' -> 'Corporate Card FAQ'."""
    return title.split(" | ")[0].strip() if title else title


def build_chunks(title, sections, source, file_type):
    """Turn [(heading, [paragraph, ...], page)] into self-contained chunk dicts."""
    chunks = []
    for heading, paragraphs, page in sections:
        paragraphs = [p for p in paragraphs if p.strip()]
        if not paragraphs:
            continue
        if heading is None:  # text before the first heading
            if sum(len(p.split()) for p in paragraphs) < MIN_INTRO_WORDS:
                continue
            heading = "Introduction"

        # Group paragraphs so no chunk body exceeds MAX_CHUNK_CHARS.
        groups, current = [], []
        for p in paragraphs:
            if current and len("\n".join(current + [p])) > MAX_CHUNK_CHARS:
                groups.append(current)
                current = []
            current.append(p)
        groups.append(current)

        for i, group in enumerate(groups, start=1):
            label = heading if len(groups) == 1 else f"{heading} (part {i} of {len(groups)})"
            chunks.append({
                "text": f"{title} - {label}\n\n" + "\n".join(group),
                "source": source,
                "title": title,
                "section": label,
                "file_type": file_type,
                "page": page or 0,
            })
    return chunks


def parse_html(path):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    h1 = soup.find("h1")
    title = (strip_site_suffix(soup.title.get_text(strip=True)) if soup.title else None) \
        or (clean(h1.get_text()) if h1 else None) or path.stem

    root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    block_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre",
                  "blockquote", "dt", "dd"]
    sections = [[None, [], None]]
    heading_path = {}
    for el in root.find_all(block_tags):
        # Skip blocks nested inside another block we already handle (e.g. <p> inside <li>).
        if el.find_parent(block_tags) is not None:
            continue
        name = el.name
        if name == "h1":
            continue  # the document title, already captured
        if name in ("h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            heading_path = {k: v for k, v in heading_path.items() if k < level}
            heading_path[level] = clean(el.get_text())
            sections.append([" > ".join(heading_path[k] for k in sorted(heading_path)), [], None])
        elif name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [clean(c.get_text()) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            sections[-1][1].append("\n".join(rows))
        elif name == "li":
            sections[-1][1].append("- " + clean(el.get_text()))
        else:
            sections[-1][1].append(clean(el.get_text()))
    return title, sections


def parse_pdf(path):
    import pymupdf

    doc = pymupdf.open(path)
    lines = []  # (size, text, page, block_id)
    for page_no, page in enumerate(doc, start=1):
        for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                text = clean("".join(s["text"] for s in line["spans"]))
                if text:
                    size = max(s["size"] for s in line["spans"])
                    lines.append((round(size, 1), text, page_no, (page_no, b_idx)))
    if not lines:
        return strip_site_suffix(doc.metadata.get("title")) or path.stem, []

    # Body text is the font size used for the most characters.
    sizes = Counter()
    for size, text, _, _ in lines:
        sizes[size] += len(text)
    body = sizes.most_common(1)[0][0]
    larger = Counter(size for size, _, _, _ in lines if size > body * 1.1)
    # Section headings: the most frequent larger size. Anything bigger is letterhead/title.
    heading_size = max(larger, key=lambda s: (larger[s], -s)) if larger else None

    title = strip_site_suffix(doc.metadata.get("title"))
    if not title:
        biggest = max(size for size, _, _, _ in lines)
        title = next(t for s, t, _, _ in lines if s == biggest)

    sections = [[None, [], 1]]
    last_block = None
    for size, text, page_no, block_id in lines:
        if size < body * 0.9 or re.fullmatch(r"Page \d+( of \d+)?", text):
            continue  # running headers/footers and page numbers
        if heading_size and size > body * 1.1:
            if size == heading_size:
                sections.append([text, [], page_no])
            continue  # letterhead and document title are skipped
        paragraphs = sections[-1][1]
        if paragraphs and block_id == last_block:
            paragraphs[-1] += " " + text  # re-join lines the PDF wrapped
        else:
            paragraphs.append(text)
        last_block = block_id
    return title, sections


def parse_docx(path):
    import docx

    d = docx.Document(path)
    title = d.core_properties.title or None
    sections = [[None, [], None]]
    for para in d.paragraphs:
        text = clean(para.text)
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style == "title":
            title = title or text
        elif style.startswith("heading"):
            if title is None and style == "heading 1":
                title = text
            else:
                sections.append([text, [], None])
        elif "list" in style:
            sections[-1][1].append("- " + text)
        else:
            sections[-1][1].append(text)
    for table in d.tables:
        rows = [" | ".join(clean(c.text) for c in row.cells) for row in table.rows]
        sections[-1][1].append("\n".join(rows))
    return title or path.stem, sections


def parse_markdown(path):
    title = None
    sections = [[None, [], None]]
    paragraph = []

    def flush():
        if paragraph:
            sections[-1][1].append(clean(" ".join(paragraph)))
            paragraph.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            if len(m.group(1)) == 1 and title is None:
                title = clean(m.group(2))
            else:
                sections.append([clean(m.group(2)), [], None])
        elif re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            flush()
            sections[-1][1].append("- " + clean(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", line)))
        elif line.strip():
            paragraph.append(line)
        else:
            flush()
    flush()
    return title or path.stem.replace("-", " ").title(), sections


def parse_text(path):
    paragraphs = [clean(p) for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace"))]
    title = path.stem.replace("-", " ").replace("_", " ").title()
    return title, [["Contents", paragraphs, None]]


PARSERS = {
    ".html": parse_html, ".htm": parse_html,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown, ".markdown": parse_markdown,
    ".txt": parse_text,
}


def document_files():
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.is_file())
    supported = [p for p in files if p.suffix.lower() in SUPPORTED]
    skipped = [p for p in files if p.suffix.lower() not in SUPPORTED]
    return supported, skipped


def chunk_all():
    supported, skipped = document_files()
    for p in skipped:
        print(f"  skipping unsupported file: {p.relative_to(DOCS_DIR)}")
    chunks = []
    for path in supported:
        source = path.relative_to(DOCS_DIR).as_posix()
        try:
            title, sections = PARSERS[path.suffix.lower()](path)
        except Exception as e:  # a bad file shouldn't stop the whole index
            print(f"  could not read {source}: {e}")
            continue
        doc_chunks = build_chunks(title, sections, source, path.suffix.lower().lstrip("."))
        print(f"  {source}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Chroma index
# ---------------------------------------------------------------------------

def fingerprint():
    """Changes whenever a document is added, removed, or edited."""
    supported, _ = document_files()
    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for p in supported:
        st = p.stat()
        h.update(f"{p.relative_to(DOCS_DIR).as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def get_client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def build_index(client):
    print(f"Indexing documents in {DOCS_DIR.relative_to(HERE.parent)} ...")
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # no existing collection
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # so similarity = 1 - distance
        metadata={"fingerprint": fingerprint()},
    )
    chunks = chunk_all()
    if chunks:
        collection.add(
            ids=[f"{c['source']}#{i}" for i, c in enumerate(chunks)],
            documents=[c["text"] for c in chunks],
            metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
        )
    print(f"Stored {len(chunks)} chunks in {DB_DIR.relative_to(HERE.parent)}\n")
    return collection


def get_collection(rebuild=False):
    client = get_client()
    if not rebuild:
        try:
            collection = client.get_collection(COLLECTION)
            if (collection.metadata or {}).get("fingerprint") == fingerprint():
                return collection
            print("Documents have changed since the last index.")
        except Exception:
            print("No index found.")
    return build_index(client)


def retrieve(collection, question):
    res = collection.query(query_texts=[question], n_results=TOP_K)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": text, "similarity": 1 - dist, **meta})
    return hits


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions from Copperline Manufacturing employees about the company's \
finance policies. You are given numbered source excerpts retrieved from Copperline's finance documents.

Rules:
- Answer only from the sources provided. Do not use outside knowledge or assumptions about \
typical company policies.
- Cite the source file for every fact, in square brackets right after it, for example \
[travel-expense-policy.html].
- If the sources don't contain the answer, or only part of it, say so plainly and say what \
is missing. Do not guess.
- Be concise and direct. Give the answer first, then any conditions or exceptions the \
sources mention."""


def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    names = " or ".join(f'"{p.name}"' for p in KEY_FILES)
    sys.exit(f"No API key found. Set ANTHROPIC_API_KEY or put the key in {names} "
             f"in {KEY_FILES[0].parent}.")


def format_sources(hits):
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f'<source index="{i}" file="{h["source"]}">\n{h["text"]}\n</source>')
    return "\n\n".join(parts)


def answer(client, question, hits):
    import anthropic

    user_content = f"<sources>\n{format_sources(hits)}\n</sources>\n\nQuestion: {question}"
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()
        print()
        if final.stop_reason == "refusal":
            print("(Claude declined to answer this question.)")
        elif final.stop_reason == "max_tokens":
            print("(Answer was cut off at the token limit.)")
    except anthropic.AuthenticationError:
        print("\nError: the API key was rejected. Check the key file.")
    except anthropic.RateLimitError:
        print("\nError: rate limited by the API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        print(f"\nAPI error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        print("\nError: could not reach the Claude API. Check your internet connection.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def ask(collection, client, question):
    hits = retrieve(collection, question)
    print(f"\nTop {len(hits)} sources:")
    for rank, h in enumerate(hits, start=1):
        preview = clean(h["text"])[:PREVIEW_CHARS]
        print(f"  {rank}. similarity {h['similarity']:.3f}  {h['source']}")
        print(f"     {preview}...")
    print("\nAnswer:")
    answer(client, question, hits)


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Answer questions from Copperline's finance documents.")
    parser.add_argument("question", nargs="*", help="the question to ask (omit for interactive mode)")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-index all documents (run after adding or removing files)")
    args = parser.parse_args()

    collection = get_collection(rebuild=args.rebuild)
    question = " ".join(args.question).strip()
    if args.rebuild and not question:
        return

    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key())

    if question:
        ask(collection, client, question)
        return

    print("Ask a question about Copperline's finance documents (blank line or Ctrl+C to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(collection, client, question)


if __name__ == "__main__":
    main()
