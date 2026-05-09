"""PSI-based drift detection simulation."""
import json, numpy as np, argparse
from pathlib import Path
np.random.seed(42)

def compute_psi(reference, current, bins=10):
    ref_hist, edges = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=edges, density=True)
    ref_hist = np.clip(ref_hist, 1e-8, None); cur_hist = np.clip(cur_hist, 1e-8, None)
    ref_pct = ref_hist / ref_hist.sum(); cur_pct = cur_hist / cur_hist.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output_dir", default="outputs"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ref = np.random.normal(50, 10, 1000)
    results = []
    for week in range(1, 13):
        drift = week * 0.5
        current = np.random.normal(50 + drift, 10 + week*0.3, 200)
        psi = compute_psi(ref, current)
        status = "OK" if psi < 0.1 else ("MONITOR" if psi < 0.25 else "RETRAIN")
        results.append({"week": week, "psi": round(psi, 4), "status": status, "mean_shift": round(drift, 1)})
        flag = "\u2705" if status == "OK" else ("\u26a0\ufe0f" if status == "MONITOR" else "\u274c")
        print(f"  Week {week:2d}: PSI={psi:.4f} {flag} {status}")
    with open(out / "drift_report.json", "w") as f: json.dump(results, f, indent=2)
    alert_week = next((r["week"] for r in results if r["status"] == "RETRAIN"), None)
    print(f"\n\u2705 Drift Detection: retrain alert at week {alert_week}")

if __name__ == "__main__": main()
