"""Portable reports: escaped HTML, no CDN, scripts, raw records, or telemetry."""
from __future__ import annotations
import html
import json
import os
import tempfile
from pathlib import Path
from .models import AuditReport


def render_html(report: AuditReport) -> str:
    def escape(value) -> str:
        return html.escape(str(value), quote=True)

    summary = report.summary
    cards = "".join(f'<div class="metric"><strong>{count}</strong><span>{escape(level)}</span></div>'
                    for level, count in summary["severity_counts"].items())
    findings = []
    for finding in report.findings:
        evidence = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in finding.evidence.items())
        findings.append(f'''<article class="finding {escape(finding.severity)}">
<div class="tags"><span>{escape(finding.severity)}</span><span>{escape(finding.confidence.replace('_', ' '))}</span></div>
<h3>{escape(finding.title)}</h3><p>{escape(finding.explanation)}</p>
<p class="fix"><strong>Next step:</strong> {escape(finding.recommendation)}</p>
<details><summary>Evidence · {escape(finding.code)}</summary><table>{evidence}</table></details></article>''')
    coverage = "".join(f'<tr><th>{escape(c.check_id)}</th><td>{escape(c.status)}</td><td>{escape(c.reason)}</td></tr>' for c in report.checks)
    measurements = "".join(f'<tr><th>{escape(check_id)}</th><td>{escape(name)}</td><td>{escape(value)}</td></tr>'
                           for check_id, metrics in report.metrics.items() for name, value in metrics.items())
    metrics_section = ('<h2>Measurements</h2><section class="panel table-wrap"><table><thead>'
                       '<tr><th>Check</th><th>Metric</th><th>Value</th></tr></thead><tbody>'
                       + measurements + '</tbody></table></section>') if measurements else ""
    manifests = "".join(f'<tr><th>{escape(split)}</th><td>{data["rows"]:,} rows · {len(data["columns"])} columns</td><td class="hash">{escape(data["sha256"])}</td></tr>' for split, data in report.datasets.items())
    body = "".join(findings) or '<article><h3>No findings from the checks that ran</h3><p>Review skipped and errored checks below before interpreting this result.</p></article>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>ProofML · Dataset audit</title><style>
:root{{color-scheme:light;--ink:#172b39;--muted:#536573;--line:#d9e2e7;--bg:#f2f6f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:40px 24px}}header{{padding:32px;border-radius:20px;background:#133c45;color:#fff}}
.brand{{font-size:15px;letter-spacing:.15em;text-transform:uppercase;color:#9ae1d1}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.15;margin:16px 0}}
header p{{max-width:750px;color:#d8eeea}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.metric{{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}}.metric strong{{display:block;font-size:32px}}.metric span{{text-transform:uppercase;font-size:12px;letter-spacing:.1em}}
article,.panel{{background:white;padding:24px;border:1px solid var(--line);border-radius:12px;margin:16px 0;overflow-wrap:anywhere}}
.finding{{border-left:5px solid #7a929e}}.critical{{border-left-color:#a62040}}.high{{border-left-color:#c34c20}}.medium{{border-left-color:#aa7a19}}
h2{{margin-top:34px}}h3{{font-size:21px;margin:12px 0}}p{{margin:10px 0}}.tags{{display:flex;gap:8px;flex-wrap:wrap}}
.tags span{{font-size:12px;background:#edf2f4;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em}}
.fix{{background:#eff8f5;padding:14px;border-radius:8px}}details{{margin-top:16px}}summary{{cursor:pointer;color:#285c65}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:12px 8px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}}
th{{font-weight:600}}.table-wrap{{overflow-x:auto}}.hash{{font-family:monospace;max-width:250px}}.note,footer{{color:var(--muted);font-size:14px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
@media(max-width:560px){{main{{padding:16px}}header{{padding:22px}}.metrics{{grid-template-columns:repeat(2,1fr)}}article,.panel{{padding:18px}}}}
@media print{{body{{background:white}}main{{padding:0}}article{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main><header><div class="brand">ProofML / Evidence before confidence</div>
<h1>Can you trust this evaluation?</h1><p>{summary['findings']} findings across {len(report.checks)} checks. Review the evidence, confirm project assumptions, and address the highest-impact issues first.</p></header>
<section class="metrics" aria-label="Finding severity counts">{cards}</section>
<p class="note">This is a dataset audit, not a model certification. Suspicious associations need investigation. No arbitrary trust score is assigned.</p>
<h2>Findings &amp; recommended actions</h2>{body}
{metrics_section}
<h2>Check coverage</h2><section class="panel table-wrap"><table><thead><tr><th>Check</th><th>Status</th><th>Reason / limitation</th></tr></thead><tbody>{coverage}</tbody></table></section>
<h2>Reproducibility manifest</h2><section class="panel table-wrap"><table><thead><tr><th>Split</th><th>Shape</th><th>Source SHA-256</th></tr></thead><tbody>{manifests}</tbody></table>
<details><summary>Audit configuration</summary><pre>{escape(json.dumps(report.config, indent=2))}</pre></details></section>
<footer>ProofML {escape(report.tool_version)} · Report schema {escape(report.schema_version)} · {escape(report.created_at)}<br>
Computed locally. Built-in checks omit raw row values; column names and aggregate statistics remain visible.</footer></main></body></html>'''


def write_reports(report: AuditReport, output: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Publish files individually with atomic rename; refuse overwrites by default.

    The pair is not a filesystem transaction. A disk failure may leave one
    completed file; neither file is exposed in a partially written state.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = (output / "report.json", output / "report.html")
    if not overwrite and any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("Report already exists; choose a new output directory or use --overwrite")
    payloads = (json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n", render_html(report))
    for target, payload in zip(targets, payloads):
        fd, temporary = tempfile.mkstemp(prefix=".proofml-", dir=output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                # Hard-link creation fails atomically if another writer won.
                os.link(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return targets
