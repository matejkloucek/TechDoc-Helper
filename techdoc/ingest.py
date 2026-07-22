import glob
import os
import re
from typing import Iterable, Iterator
import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

MIN_CHUNK_CHARS = 50

# Best-effort structural extractors. These are deliberately regex heuristics,
# not a markdown AST or a Python parser: retrieval is fuzzy, so the cost/benefit
# favours simple, robust extraction over exactness. (A principled upgrade for
# docs would be MarkdownHeaderTextSplitter, which attaches header hierarchy as
# metadata directly -- noted as a future improvement.)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", re.MULTILINE)
_SYMBOL_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)
# These MDX docs declare their page title in YAML frontmatter, not as an H1.
_FRONTMATTER_TITLE_RE = re.compile(
    r"\A---\s*\n.*?^title:\s*(.+?)\s*$.*?\n---\s*\n", re.DOTALL | re.MULTILINE
)

def load_manifest(path="data/corpus_manifest.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def is_excluded(file_path: str, excludes: list[str]) -> bool:
    """Check if the file_path matches any of the exclude patterns."""
    for pattern in excludes:
        if glob.fnmatch.fnmatch(file_path, pattern):
            return True
    return False
    
def iter_source_files(manifest: dict) -> Iterator[dict]:
    raw_root = manifest["raw_root"]
    excludes = manifest["global_excludes"]
    for repo_name, repo in manifest["repos"].items():
      commit = repo["commit"]
      for entry in repo["entries"]:
          for file_path in glob.glob(os.path.join(raw_root, entry["path"]), recursive=True):
              if is_excluded(file_path, excludes):
                  continue
              yield { "path": file_path, "type": entry.get("type"), "language": entry.get("language"),
                      "source_repo": repo_name, "commit": commit }
              

def get_splitter(record_type: str) -> RecursiveCharacterTextSplitter:
      """Return a splitter tuned for prose vs code, based on record['type']."""
      if record_type == "doc":
        return RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, chunk_size=1000, chunk_overlap=150)
      elif record_type == "code":
          return RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=1000, chunk_overlap=150)
      else:
          raise ValueError(f"Unknown record type: {record_type}")

def read_text(path: str) -> str:
    """Read text from a file, handling encoding issues."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()
        

def relpath_in_repo(path: str, raw_root: str, source_repo: str) -> str:
    """Path relative to the repo root, e.g. 'src/oss/langgraph/streaming.mdx'.

    Nicer for citations than the on-disk absolute/relative path, and stable
    across machines. `raw_root/source_repo` is the repo checkout root.
    """
    repo_root = os.path.join(raw_root, source_repo)
    return os.path.relpath(path, repo_root)


def module_path(rel_path: str) -> str:
    """Dotted Python module path from a repo-relative .py path, best-effort.

    We anchor on the top-level package name (the segment that also names the
    package dir), so '.../libs/core/langchain_core/messages/ai.py' ->
    'langchain_core.messages.ai'. If no known anchor is found we fall back to
    joining the trailing path segments, which is still consistent per file.
    """
    parts = Path(rel_path).with_suffix("").parts
    anchors = ("langchain_core", "langgraph")
    start = 0
    for i, p in enumerate(parts):
        if p in anchors:
            start = i  # last match wins: the package dir, not the repo/libs dir
    dotted = ".".join(parts[start:])
    # __init__ adds no information to the module name
    return dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted


def file_title(text: str, fallback: str) -> str:
    """Page title: prefer YAML frontmatter `title:`, then first H1/heading, else fallback."""
    fm = _FRONTMATTER_TITLE_RE.search(text)
    if fm:
        return fm.group(1).strip().strip("\"'")
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else fallback


def last_heading_in(chunk: str) -> str | None:
    """Nearest heading contained in this chunk (last one wins), else None."""
    matches = _HEADING_RE.findall(chunk)
    return matches[-1].strip() if matches else None


def last_symbol_in(chunk: str) -> str | None:
    """Nearest def/class name contained in this chunk (last one wins)."""
    matches = _SYMBOL_RE.findall(chunk)
    return matches[-1] if matches else None


def chunk_records(records: Iterable[dict], raw_root: str = "data/raw") -> list[Document]:
    """Read each source file and split it into Documents with enriched metadata.

    Per-file metadata (title/module/rel_path) is computed once; per-chunk
    metadata (heading/symbol/chunk_id) is computed inside the loop.
    """
    documents: list[Document] = []
    for record in records:
        text = read_text(record["path"])
        splitter = get_splitter(record["type"])

        # --- per-file metadata, computed once ---
        rel_path = relpath_in_repo(record["path"], raw_root, record["source_repo"])
        base = {
            "path": record["path"],
            "rel_path": rel_path,
            "type": record["type"],
            "language": record["language"],
            "source_repo": record["source_repo"],
            "commit": record["commit"],
        }
        if record["type"] == "doc":
            base["title"] = file_title(text, fallback=Path(rel_path).name)
        else:  # code
            base["module"] = module_path(rel_path)

        # split_text returns a list[str] of chunk strings
        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks):
            content = chunk.strip()
            if len(content) < MIN_CHUNK_CHARS:
                continue

            metadata = dict(base)
            # stable, deterministic id -> reproducible evals & citation handles
            metadata["chunk_id"] = f"{record['source_repo']}:{rel_path}:{index}"

            # --- per-chunk structural context (best-effort) ---
            if record["type"] == "doc":
                metadata["heading"] = last_heading_in(chunk) or base["title"]
            else:  # code
                metadata["symbol"] = last_symbol_in(chunk)

            documents.append(Document(page_content=content, metadata=metadata))
    return documents



