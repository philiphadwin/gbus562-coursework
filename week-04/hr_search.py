"""Search Hartwell Foods' HR documents (the retrieval half of week-03/ask.py).

The HTML parsing, section chunking, Chroma index and fingerprint-based rebuild are adapted
from week-03/ask.py. Differences:
  - Only HTML is supported, because every HR document is HTML.
  - There is no Claude call here. search() returns the closest chunks, and the AI app that
    calls the MCP tool writes the answer and cites the files. So no API key is needed.
  - Nothing is printed to stdout. An MCP server running over stdio uses stdout for protocol
    messages, so progress goes to the log (stderr) instead.
"""

import contextlib
import hashlib
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "hartwell-hr" / "hr-docs"
DB_DIR = HERE / "chroma_db"   # ignored by Git (.gitignore has chroma_db/)
COLLECTION = "hartwell_hr_docs"
TOP_K = 4
MAX_CHUNK_CHARS = 1500        # sections longer than this are split on paragraph boundaries
MIN_INTRO_WORDS = 25          # text before the first heading is kept only if it's this long
CHUNKER_VERSION = "1"         # bump to force a rebuild when the chunking logic changes

log = logging.getLogger("hr_search")


# ---------------------------------------------------------------------------
# Chunking (from week-03/ask.py)
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_site_suffix(title):
    """'Remote Work Policy | Hartwell Foods' -> 'Remote Work Policy'."""
    return title.split(" | ")[0].strip() if title else title


def build_chunks(title, sections, source):
    """Turn [(heading, [paragraph, ...])] into self-contained chunk dicts."""
    chunks = []
    for heading, paragraphs in sections:
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
    sections = [[None, []]]
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
            sections.append([" > ".join(heading_path[k] for k in sorted(heading_path)), []])
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


def document_files():
    return sorted(p for p in DOCS_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in (".html", ".htm"))


def chunk_all():
    chunks = []
    for path in document_files():
        source = path.relative_to(DOCS_DIR).as_posix()
        try:
            title, sections = parse_html(path)
        except Exception as e:  # a bad file shouldn't stop the whole index
            log.warning("could not read %s: %s", source, e)
            continue
        doc_chunks = build_chunks(title, sections, source)
        log.info("  %s: %d chunks", source, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Chroma index (from week-03/ask.py)
# ---------------------------------------------------------------------------

def fingerprint():
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for p in document_files():
        st = p.stat()
        h.update(f"{p.relative_to(DOCS_DIR).as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def _build(client):
    log.info("Indexing HR documents in %s ...", DOCS_DIR)
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
    log.info("Stored %d chunks in %s", len(chunks), DB_DIR)
    return collection


def get_collection(rebuild=False):
    """Open the index, rebuilding it if asked or if the documents have changed."""
    # Keep any library output off stdout, which belongs to the MCP protocol.
    with contextlib.redirect_stdout(sys.stderr):
        client = _client()
        if not rebuild:
            try:
                collection = client.get_collection(COLLECTION)
                if (collection.metadata or {}).get("fingerprint") == fingerprint():
                    return collection
                log.info("HR documents have changed since the last index.")
            except Exception:
                log.info("No index found.")
        return _build(client)


def search(collection, query, top_k=TOP_K):
    """Return the top_k chunks closest to the query, best first."""
    n = min(top_k, collection.count())
    if n == 0:
        return []
    with contextlib.redirect_stdout(sys.stderr):
        res = collection.query(query_texts=[query], n_results=n)
    return [
        {"source": meta["source"], "title": meta["title"], "section": meta["section"],
         "similarity": round(1 - dist, 3), "text": text}
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
