"""Data-drift report with Evidently and a simple alert rule.

Evidently picks a per-column test from column type and sample size (K-S for
small numerical samples, chi-squared for categorical, Wasserstein / Jensen-Shannon
for large ones). A column drifts if its p-value is below the threshold (p-value
tests) or its distance is at or above it (distance tests). An alert fires when the
share of drifted columns exceeds ``share_threshold``.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], share_threshold: float = 0.2,
                 out_dir: Path | None = None) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset(columns=columns, drift_share=share_threshold)]).run(
        reference_data=reference[columns], current_data=current[columns]
    )
    raw = snapshot.dict()
    per_column = {}
    share = None
    for m in raw["metrics"]:
        cfg = m["config"]
        if cfg.get("type", "").endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
        elif cfg.get("type", "").endswith("ValueDrift"):
            method, threshold, value = cfg["method"], float(cfg["threshold"]), float(m["value"])
            drifted = value < threshold if "p_value" in method else value >= threshold
            per_column[cfg["column"]] = {"method": method, "value": round(value, 6), "threshold": threshold, "drifted": drifted}
    drifted = sorted(c for c, v in per_column.items() if v["drifted"])
    share = share if share is not None else len(drifted) / max(1, len(per_column))
    result = {
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "drift_share": round(share, 4),
        "share_threshold": share_threshold,
        "drifted_columns": drifted,
        "alert": share > share_threshold,
        "columns": per_column,
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out_dir / "drift_report.html"))
        (out_dir / "drift_summary.json").write_text(json.dumps(result, indent=2))
    return result


def send_alert(result: dict, webhook_url: str | None = None) -> bool:
    """Log the alert; POST it to a Slack-compatible webhook if ``DRIFT_WEBHOOK_URL`` is set."""
    if not result.get("alert"):
        return False
    msg = f"Data drift alert: {result['drift_share']:.0%} of columns drifted (threshold {result['share_threshold']:.0%}): {', '.join(result['drifted_columns'])}"
    log.warning(msg)
    url = webhook_url or os.environ.get("DRIFT_WEBHOOK_URL")
    if url:
        req = urllib.request.Request(url, data=json.dumps({"text": msg}).encode(), headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=10)  # noqa: S310 - operator-configured URL
    return True
