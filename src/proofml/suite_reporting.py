"""Escaped, offline suite summaries; never concatenate untrusted HTML content."""
from html import escape


def render_suite(suite):
    def esc(value):
        return escape(str(value), quote=True)
    sections = []
    for name, report in suite.reports.items():
        rows = []
        for check in report.checks:
            coverage = check.coverage
            counts = f"{coverage.evaluated}/{coverage.total} {coverage.unit}" if coverage else "Not declared"
            metrics = "; ".join(f"{key}: {value:.6g}" for key, value in check.metrics.items()) or "No measurements"
            rows.append(f"<tr><th>{esc(check.check_id)}</th><td>{esc(check.status)}</td><td>{esc(counts)}</td><td>{esc(metrics)}</td></tr>")
        findings = "".join(f"<li>{esc(f.severity)} / {esc(f.code)}: {esc(f.title)}<br>{esc(f.recommendation)}</li>" for f in report.findings)
        reasons = "".join(f"<li>{esc(c.check_id)}: {esc(c.reason)}</li>" for c in report.checks if c.reason)
        sections.append(f"<section><h2>{esc(name)}</h2><p>{esc(report)}</p><div class='table'><table><thead><tr><th>Check</th><th>Status</th><th>Assessed population</th><th>Measurements</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><ul>{findings}</ul><details><summary>Check limitations</summary><ul>{reasons}</ul></details></section>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>ProofML · Evaluation suite</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f2f6f7;color:#172b39;margin:0}}main{{max-width:1200px;margin:auto;padding:28px}}
header{{background:#133c45;color:white;padding:28px;border-radius:14px}}section{{background:white;padding:22px;margin-top:20px;border:1px solid #d9e2e7;border-radius:12px;overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d9e2e7;vertical-align:top}}.table{{overflow-x:auto}}details{{margin-top:14px}}
</style></head><body><main><header><h1>Evaluation across slices and folds</h1><p>{esc(suite)}</p></header>
<p>Each report retains its own population and metric definitions. Overlapping slices are not independent and their counts must not be summed. This summary is not a release decision; apply a SuitePolicy to enforce requirements.</p>
{''.join(sections)}<footer>Computed locally. Report names are visible; do not use personal identifiers as slice names. Full provenance is in suite.json.</footer></main></body></html>"""
