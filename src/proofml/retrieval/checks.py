"""Framework-neutral retrieval evaluation with explicit binary relevance semantics."""
import math

from ..models import CheckResult, Coverage, Finding
from .data import RetrievalContext


class RetrievalInputsCheck:
    id = "retrieval_inputs"

    def run(self, ctx: RetrievalContext) -> CheckResult:
        rankings, judgments = ctx.data.retrieved, ctx.data.relevant
        metrics = {
            "queries": len(rankings),
            "empty_result_queries": sum(not ranking for ranking in rankings),
            "unjudged_queries": sum(not truth for truth in judgments),
            "duplicate_result_queries": sum(len(ranking) != len(set(ranking)) for ranking in rankings),
            "short_result_queries": sum(len(ranking) < ctx.config.k for ranking in rankings),
        }
        findings = []
        for key, code, title, severity, explanation, recommendation in (
            ("empty_result_queries", "empty_retrieval_results", "Queries returned no results", "medium",
             "Some queries have an empty ranked list.", "Review filters, indexing coverage, and intentional abstention behavior."),
            ("unjudged_queries", "missing_relevance_judgments", "Queries lack positive relevance judgments", "medium",
             "Queries with no known relevant IDs are excluded from all ranking averages, not counted as successes.",
             "Add relevance judgments or evaluate unanswerable queries separately; this module cannot infer relevance."),
            ("duplicate_result_queries", "duplicate_retrieval_results", "Ranked results repeat document IDs", "high",
             "Some rankings contain repeated IDs; repetitions occupy rank positions but earn no additional relevance credit.",
             "Review merging and reranking; deduplicate using stable document or chunk identifiers."),
        ):
            if metrics[key]:
                findings.append(Finding(code, severity, "confirmed", title, explanation, recommendation,
                                        evidence={"affected_queries": metrics[key]}))
        return CheckResult.complete(self.id, findings, metrics=metrics,
                                    coverage=Coverage(len(rankings), len(rankings), "queries"))


class RetrievalCorpusCheck:
    id = "retrieval_corpus"

    def run(self, ctx: RetrievalContext) -> CheckResult:
        corpus = ctx.data.corpus_ids
        if corpus is None:
            return CheckResult(self.id, "skipped", reason="Supply corpus_ids to verify ID membership in the evaluated corpus.")
        findings = []
        for code, groups, title in (
            ("unknown_retrieved_ids", ctx.data.retrieved, "Retrieved IDs are absent from the corpus"),
            ("unreachable_relevant_ids", ctx.data.relevant, "Relevant IDs are absent from the corpus"),
        ):
            counts = [sum(value not in corpus for value in group) for group in groups]
            if any(counts):
                findings.append(Finding(code, "high", "confirmed", title,
                    "Some supplied document IDs are not members of the declared corpus snapshot.",
                    "Align corpus versions and document/chunk ID namespaces before comparing retrievers.",
                    evidence={"affected_queries": sum(n > 0 for n in counts), "missing_id_occurrences": sum(counts)}))
        return CheckResult.complete(self.id, findings, coverage=Coverage(len(ctx.data.retrieved), len(ctx.data.retrieved), "queries"))


class RetrievalRankingCheck:
    id = "retrieval_ranking"

    def run(self, ctx: RetrievalContext) -> CheckResult:
        k = ctx.config.k
        totals = {f"{name}@{k}": 0.0 for name in ("precision", "recall", "hit_rate", "mrr", "ndcg")}
        assessed = 0
        for ranking, relevant in zip(ctx.data.retrieved, ctx.data.relevant):
            if not relevant:
                continue
            assessed += 1
            seen = set()
            hits, reciprocal, dcg = 0, 0.0, 0.0
            for rank, identifier in enumerate(ranking[:k], 1):
                if identifier in relevant and identifier not in seen:
                    hits += 1
                    dcg += 1 / math.log2(rank + 1)
                    if not reciprocal:
                        reciprocal = 1 / rank
                seen.add(identifier)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            # Precision always divides by k: an empty/short result list cannot
            # obtain perfect precision just by returning fewer documents.
            totals[f"precision@{k}"] += hits / k
            totals[f"recall@{k}"] += hits / len(relevant)
            totals[f"hit_rate@{k}"] += bool(hits)
            totals[f"mrr@{k}"] += reciprocal
            totals[f"ndcg@{k}"] += dcg / ideal
        coverage = Coverage(len(ctx.data.retrieved), assessed, "queries")
        if not assessed:
            return CheckResult(self.id, "skipped", reason="No query has positive relevance judgments; ranking metrics and thresholds are unassessed.", coverage=coverage)
        metrics = {key: value / assessed for key, value in totals.items()}
        metrics.update({"evaluated_queries": assessed, "excluded_queries": len(ctx.data.retrieved) - assessed})
        findings = []
        for name, threshold in (("recall", ctx.config.min_recall), ("ndcg", ctx.config.min_ndcg)):
            if threshold is not None and metrics[f"{name}@{k}"] < threshold:
                findings.append(Finding(f"retrieval_{name}_below_threshold", "high", "confirmed", f"Retrieval {name} below declared minimum",
                    "The macro-average ranking metric is below the user-supplied requirement on queries with positive judgments.",
                    "Inspect indexing, chunking, retrieval filters, ranking, and judgment coverage before deployment.",
                    evidence={"metric": f"{name}@{k}", "value": metrics[f"{name}@{k}"], "threshold": threshold,
                              "evaluated_queries": assessed}))
        result = CheckResult.complete(self.id, findings, metrics=metrics)
        return CheckResult(self.id, result.status, result.findings,
            reason="Binary judgments; unlisted IDs count as nonrelevant; macro average over queries with positives. Duplicate hits receive credit once. This does not assess generated answers.", metrics=metrics, coverage=coverage)


def default_retrieval_checks():
    return (RetrievalInputsCheck(), RetrievalCorpusCheck(), RetrievalRankingCheck())
