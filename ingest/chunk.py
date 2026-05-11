"""Split parsed PDF sections into token-bounded chunks with metadata."""

import logging
import re
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 400
MAX_TOKENS = 700


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_at_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries ('. '), falling back to token-count splits."""
    parts = re.split(r"(?<=\.) ", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        # No sentence boundaries found — hard-split by token count
        tokens = _ENCODER.encode(text)
        result = []
        for i in range(0, len(tokens), MAX_TOKENS):
            result.append(_ENCODER.decode(tokens[i : i + MAX_TOKENS]))
        return result
    return parts


def _chunk_text(text: str) -> list[str]:
    """Split `text` into chunks of MIN–MAX tokens at paragraph boundaries.

    Args:
        text: Raw section text.

    Returns:
        List of chunk strings each within [MIN_TOKENS, MAX_TOKENS] tokens
        (except possibly the last chunk if the section is short).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Expand oversized paragraphs into sentence-level pieces
    pieces: list[str] = []
    for para in paragraphs:
        if _count_tokens(para) > MAX_TOKENS:
            pieces.extend(_split_at_sentences(para))
        else:
            pieces.append(para)

    # Greedily merge pieces into chunks
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _count_tokens(piece)
        if current_tokens + piece_tokens > MAX_TOKENS and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [piece]
            current_tokens = piece_tokens
        else:
            current_parts.append(piece)
            current_tokens += piece_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Merge underflow chunks (< MIN_TOKENS) with the next chunk
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        if _count_tokens(chunks[i]) < MIN_TOKENS and i + 1 < len(chunks):
            combined = chunks[i] + "\n\n" + chunks[i + 1]
            chunks[i + 1] = combined
            i += 1
        else:
            merged.append(chunks[i])
            i += 1

    return merged if merged else chunks


def chunk_sections(sections: list[dict], metadata: dict[str, Any]) -> list[dict]:
    """Chunk all sections of one paper into token-bounded dicts.

    Args:
        sections: List of section dicts from parse_pdfs.parse_pdf().
        metadata: Paper-level metadata dict (must include arxiv_id, title, authors, url).

    Returns:
        List of chunk dicts with all required fields.
    """
    arxiv_id = metadata.get("arxiv_id", "unknown")
    title = metadata.get("title", "")
    authors = metadata.get("authors", [])
    url = metadata.get("url", "")

    all_chunks: list[dict] = []
    chunk_index = 0

    for section in sections:
        texts = _chunk_text(section["text"])
        for text in texts:
            chunk_dict = {
                "chunk_id": f"{arxiv_id}_chunk_{chunk_index:04d}",
                "text": text,
                "arxiv_id": arxiv_id,
                "section": section["section"],
                "token_count": _count_tokens(text),
                "chunk_index": chunk_index,
                "page_start": section.get("page_start", 1),
                "page_end": section.get("page_end", 1),
                "title": title,
                "authors": authors,
                "url": url,
            }
            all_chunks.append(chunk_dict)
            chunk_index += 1

    return all_chunks


def chunk_all_papers(
    parsed: dict[str, list[dict]],
    metadata_map: dict[str, dict],
) -> list[dict]:
    """Chunk all papers' sections.

    Args:
        parsed: Dict mapping arxiv_id → list of section dicts.
        metadata_map: Dict mapping arxiv_id → paper metadata dict.

    Returns:
        Flat list of all chunk dicts across all papers.
    """
    all_chunks: list[dict] = []
    for arxiv_id, sections in parsed.items():
        meta = metadata_map.get(arxiv_id, {"arxiv_id": arxiv_id})
        chunks = chunk_sections(sections, meta)
        all_chunks.extend(chunks)
        logger.info("Chunked %s: %d chunks from %d sections", arxiv_id, len(chunks), len(sections))
    return all_chunks


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Chunk parsed PDF sections")
    parser.add_argument("--parsed-path", default="data/parsed_sections.json")
    parser.add_argument("--metadata-path", default="data/pdfs/metadata.json")
    parser.add_argument("--output", default="data/chunks.json")
    args = parser.parse_args()

    with open(args.parsed_path) as f:
        parsed = json.load(f)
    with open(args.metadata_path) as f:
        metadata_map = json.load(f)

    chunks = chunk_all_papers(parsed, metadata_map)
    print(f"Created {len(chunks)} chunks from {len(parsed)} papers")
    with open(args.output, "w") as f:
        json.dump(chunks, f, indent=2)
