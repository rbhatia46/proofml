"""Bounded document ingestion. No text, label, or per-document hash in reports."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
import json
import unicodedata

from .._validation import ordered_iterable
from .config import TextConfig


@dataclass(frozen=True)
class TextDataset:
    documents: tuple[str, ...]
    labels: tuple[str | None, ...] | None
    fingerprint: str
    size_bytes: int

    def metadata(self):
        return {"rows": len(self.documents), "columns": ["text"] + (["label"] if self.labels is not None else []),
                "sha256": self.fingerprint, "format": "text", "size_bytes": self.size_bytes}


def load_text(documents, labels, config: TextConfig) -> TextDataset:
    docs = ordered_iterable(documents, "documents")
    label_iter = ordered_iterable(labels, "labels") if labels is not None else None
    sentinel = object()
    pairs = zip_longest(docs, label_iter, fillvalue=sentinel) if label_iter is not None else ((doc, None) for doc in docs)
    normalized, encoded_labels = [], []
    digest = sha256(b"proofml:text:v1\n")
    size = 0
    for document, label in pairs:
        if document is sentinel or label is sentinel:
            raise ValueError("Labels must match the document count exactly")
        if len(normalized) >= config.max_documents:
            raise ValueError("Text input exceeds max_documents; no sampling was performed")
        if not isinstance(document, str):
            raise TypeError("Each document must be a string; use an empty string for a missing document")
        if label is not None and type(label) not in {str, int}:
            raise TypeError("Text labels must be strings, integers, or None")
        if len(document) > config.max_bytes or (isinstance(label, str) and len(label) > config.max_bytes):
            raise ValueError("Text input exceeds max_bytes")
        payload = json.dumps([document, label], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        size += len(payload)
        if size > config.max_bytes:
            raise ValueError("Text input exceeds max_bytes; no sampling was performed")
        digest.update(payload + b"\n")
        # Keep missing labels distinct from class names; integer 1 and string
        # '1' remain distinct. Normalization applies to documents, not labels.
        encoded_labels.append(None if label is None or label == "" else json.dumps([type(label).__name__, label]))
        if config.normalization == "unicode":
            document = " ".join(unicodedata.normalize("NFKC", document).casefold().split())
        normalized.append(document)
    if not normalized:
        raise ValueError("Text input must contain at least one document")
    return TextDataset(tuple(normalized), tuple(encoded_labels) if labels is not None else None, digest.hexdigest(), size)


@dataclass(frozen=True)
class TextContext:
    train: TextDataset
    test: TextDataset | None
    config: TextConfig
