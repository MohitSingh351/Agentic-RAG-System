"""Fetch recent arXiv cs.AI papers and download PDFs with metadata."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv

logger = logging.getLogger(__name__)


def fetch_papers(
    max_results: int = 100,
    days_back: int = 90,
    output_dir: str = "data/pdfs",
) -> list[dict]:
    """Fetch cs.AI papers from arXiv published within the last `days_back` days.

    Args:
        max_results: Maximum number of papers to fetch from the API.
        days_back: Only keep papers published within this many days.
        output_dir: Directory to save PDFs and metadata.json.

    Returns:
        List of metadata dicts for successfully downloaded papers.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers: list[dict] = []
    existing_metadata: dict = {}
    metadata_path = out_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing_metadata = json.load(f)

    for entry in client.results(search):
        published = entry.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            logger.debug("Skipping %s — published %s before cutoff", entry.entry_id, published.date())
            continue

        arxiv_id = entry.entry_id.split("/abs/")[-1].split("v")[0]
        pdf_path = out_path / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            try:
                entry.download_pdf(dirpath=str(out_path), filename=f"{arxiv_id}.pdf")
                logger.info("Downloaded %s", arxiv_id)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", arxiv_id, exc)
                time.sleep(3)
                continue
            time.sleep(3)

        meta = {
            "arxiv_id": arxiv_id,
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "abstract": entry.summary,
            "published": published.date().isoformat(),
            "url": entry.entry_id,
            "pdf_path": str(pdf_path),
        }
        papers.append(meta)
        existing_metadata[arxiv_id] = meta

    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)

    logger.info("Fetched %d papers, saved metadata to %s", len(papers), metadata_path)
    return papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch arXiv cs.AI papers")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--output-dir", default="data/pdfs")
    args = parser.parse_args()

    results = fetch_papers(args.max_results, args.days_back, args.output_dir)
    print(f"Fetched {len(results)} papers")
