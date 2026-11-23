"""One-page weekly summary for the CVM team: plan, money (assumption-based), monitoring decision.

    python -m ope.monitoring.weekly_report --snapshot-date 2026-11-09
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from ope import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date", default=date.today().isoformat())
    d = ap.parse_args().snapshot_date
    rep = config.path("reports")
    run = json.loads((rep / "last_batch_run.json").read_text())
    mon_path = rep / "monitoring" / f"weekly_{d}.json"
    mon = json.loads(mon_path.read_text()) if mon_path.exists() else None
    p = run["policy"]
    lines = [
        f"# Weekly plan {d}",
        "",
        f"- Subscribers scored: {run['rows']:,}",
        f"- Contacts planned: {p['contacts']:,}; spend {p['spend']:,.0f} EGP; expected value {p['expected_value']:,.0f} EGP (assumption-based)",
    ]
    for a, v in p["per_action"].items():
        lines.append(f"  - {a}: {v['n']:,} contacts, EV {v['ev']:,.0f} EGP")
    lines.append(f"- Models: {', '.join(run['model_versions'].values())}")
    if mon:
        dec = mon["decision"]
        lines.append(f"- Monitoring: {'RETRAIN requested: ' + '; '.join(dec['reasons']) if dec['retrain'] else 'no retrain trigger'}")
        for t, c in mon["cbpe"].items():
            lines.append(f"  - {t}: estimated AUC {c['estimated_auc']:.4f} (reference {c['reference_auc']:.4f})")
    out = rep / "weekly"
    out.mkdir(exist_ok=True)
    (out / f"{d}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
