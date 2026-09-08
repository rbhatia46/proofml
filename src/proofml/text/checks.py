"""Document checks: deterministic observations, not semantic leakage proofs."""
from collections import Counter, defaultdict
import re

from ..models import CheckResult, Coverage, Finding
from .data import TextContext


class _TokenBudgetExceeded(Exception):
    pass


class _ShingleBudgetExceeded(Exception):
    pass


class TextQualityCheck:
    id = "text_quality"

    def run(self, ctx: TextContext) -> CheckResult:
        findings = []
        for name, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None:
                continue
            empty = sum(not doc.strip() for doc in data.documents)
            counts = Counter(doc for doc in data.documents if doc.strip())
            repeated = sum(n - 1 for n in counts.values())
            for code, count, title, recommendation in (
                ("empty_documents", empty, "Empty documents", "Review extraction failures and intentional empty-input handling."),
                ("duplicate_documents", repeated, "Repeated documents", "Review duplicates and sampling weights; repeated observations can be intentional."),
            ):
                if count:
                    findings.append(Finding(code, "medium", "confirmed", f"{title} in {name}",
                        "Observed under the configured document normalization; raw documents are not included.",
                        recommendation, evidence={"split": name, "affected_rows": count}))
        rows = len(ctx.train.documents) + (len(ctx.test.documents) if ctx.test is not None else 0)
        return CheckResult.complete(self.id, findings, coverage=Coverage(rows, rows, "documents"))


class TextLabelsCheck:
    id = "text_labels"

    def run(self, ctx: TextContext) -> CheckResult:
        if ctx.train.labels is None:
            return CheckResult(self.id, "skipped", reason="Supply y to assess single-label classification consistency.")
        findings = []
        combined = defaultdict(set)
        for name, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None or data.labels is None:
                continue
            missing = data.labels.count(None)
            if missing:
                findings.append(Finding("missing_text_labels", "high", "confirmed", f"Missing labels in {name}",
                    "Some supplied single-label classification targets are absent.", "Provide labels or explicitly exclude unlabelled observations from supervised evaluation.",
                    evidence={"split": name, "affected_rows": missing}))
            for document, label in zip(data.documents, data.labels):
                if document.strip() and label is not None:
                    combined[document].add(label)
        conflicts = sum(len(labels) > 1 for labels in combined.values())
        if conflicts:
            findings.append(Finding("conflicting_text_labels", "high", "needs_context", "Equivalent documents have different labels",
                "Under configured normalization, the same document has multiple single-label targets across the supplied splits.",
                "Review annotation policy and normalization; context-dependent or multi-label targets require a different contract.",
                evidence={"conflicting_document_groups": conflicts}))
        if ctx.test is not None and ctx.test.labels is not None:
            known = set(ctx.train.labels) - {None}
            unseen = [label for label in ctx.test.labels if label is not None and label not in known]
            if unseen:
                findings.append(Finding("unseen_text_labels", "high", "confirmed", "Test classes absent from training",
                    "Some observed test classes have no labelled training examples.", "Check label mapping and closed-set evaluation coverage.",
                    evidence={"unseen_classes": len(set(unseen)), "affected_rows": len(unseen)}))
        return CheckResult.complete(self.id, findings)


class TextOverlapCheck:
    id = "text_overlap"

    def run(self, ctx: TextContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test corpus is required.")
        known = {doc for doc in ctx.train.documents if doc.strip()}
        overlapping = [doc for doc in ctx.test.documents if doc.strip() and doc in known]
        findings = []
        if overlapping:
            findings.append(Finding("text_train_test_overlap", "high", "confirmed", "Test documents occur in training",
                "Document equality is established under the configured normalization, not necessarily byte equality or improper leakage.",
                "Review split construction, templated inputs, and document-level grouping.",
                evidence={"affected_test_rows": len(overlapping), "distinct_documents": len(set(overlapping)),
                          "normalization": ctx.config.normalization}))
        return CheckResult.complete(self.id, findings)


class TextNearDuplicateCheck:
    id = "text_near_duplicates"

    def run(self, ctx: TextContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test corpus is required.")
        size = ctx.config.shingle_size
        total_shingles = 0

        def shingles(document):
            nonlocal total_shingles
            tokens = []
            for match in re.finditer(r"\w+", document, flags=re.UNICODE):
                if len(tokens) >= ctx.config.max_tokens_per_document:
                    raise _TokenBudgetExceeded
                tokens.append(match.group())
            terms = set()
            for i in range(len(tokens) - size + 1):
                term = tuple(tokens[i:i + size])
                if term not in terms:
                    total_shingles += 1
                    if total_shingles > ctx.config.max_shingles:
                        raise _ShingleBudgetExceeded
                    terms.add(term)
            return frozenset(terms)

        try:
            # Collapse identical training documents: duplicate rows should not
            # multiply candidate work. Exact overlap is a separate linear check.
            train = {doc: shingles(doc) for doc in dict.fromkeys(ctx.train.documents) if doc.strip()}
            test = {doc: shingles(doc) for doc in dict.fromkeys(ctx.test.documents) if doc.strip()}
        except _TokenBudgetExceeded:
            return CheckResult(self.id, "skipped", reason="Token budget exceeded; no partial near-duplicate result is reported. Increase max_tokens_per_document or supply shorter documents.")
        except _ShingleBudgetExceeded:
            return CheckResult(self.id, "skipped", reason="Shingle memory budget exceeded; no partial near-duplicate result is reported. Increase max_shingles or audit smaller corpora.")
        index = defaultdict(list)
        for doc, terms in train.items():
            for term in terms:
                index[term].append(doc)
        if not index or not any(test.values()):
            return CheckResult(self.id, "skipped", reason="No eligible document pair: near-duplicate comparison needs at least shingle_size word tokens in both splits.")
        visits = 0
        affected = set()
        for document, terms in test.items():
            if document in train or not terms:
                continue
            candidates = set()
            for term in sorted(terms):
                for candidate in index.get(term, ()):
                    visits += 1
                    if visits > ctx.config.max_candidate_visits:
                        return CheckResult(self.id, "skipped", reason="Candidate budget exceeded; no partial near-duplicate result is reported. Exact overlap still runs. Increase max_candidate_visits or audit smaller corpora.")
                    candidates.add(candidate)
            for candidate in candidates:
                other = train[candidate]
                similarity = len(terms & other) / len(terms | other)
                if similarity >= ctx.config.near_duplicate_threshold:
                    affected.add(document)
                    break
        findings = []
        if affected:
            findings.append(Finding("text_near_train_test_overlap", "high", "suspicious", "Test documents resemble training documents",
                "Word-shingle Jaccard similarity reaches the configured threshold. This is lexical similarity, not semantic equivalence.",
                "Review shared templates and duplicated passages; group related source documents before splitting.",
                evidence={"affected_test_rows": sum(doc in affected for doc in ctx.test.documents),
                          "distinct_test_documents": len(affected), "threshold": ctx.config.near_duplicate_threshold,
                          "shingle_size": size}))
        result = CheckResult.complete(self.id, findings, metrics={
            "eligible_unique_train_documents": sum(bool(v) for v in train.values()),
            "eligible_unique_test_documents": sum(bool(v) for v in test.values()), "candidate_visits": visits})
        return CheckResult(self.id, result.status, result.findings,
            reason="Lexical comparison only; short/wordless documents are excluded and exact matches are handled by text_overlap. Not a language-aware tokenizer.", metrics=result.metrics)


def default_text_checks():
    return (TextQualityCheck(), TextLabelsCheck(), TextOverlapCheck(), TextNearDuplicateCheck())
