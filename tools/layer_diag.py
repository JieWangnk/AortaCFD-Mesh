#!/usr/bin/env python3
# tools/layer_diag.py
import re
import math
import argparse
from pathlib import Path

SNAPPY_NAMES = {
    "nSurfaceLayers": r"\bnSurfaceLayers\s+(\d+)",
    "expansionRatio": r"\bexpansionRatio\s+([0-9.]+)",
    # relative mode (if present)
    "relativeSizes": r"\brelativeSizes\s+(true|false)",
    "firstLayerThickness_rel": r"\bfirstLayerThickness\s+([0-9.eE+-]+)\b",
    "finalLayerThickness_rel": r"\bfinalLayerThickness\s+([0-9.eE+-]+)\b",
    # absolute mode (if present)
    "firstLayerThickness_abs": r"\bfirstLayerThickness\s+([0-9.eE+-]+)\b",
    "minThickness_abs": r"\bminThickness\s+([0-9.eE+-]+)\b",
}

LOG_PATTERNS = [
    # Your optimizer's line:
    r"Final layer metrics:\s*[0-9.]+\s*layers,\s*([0-9.]+)\s*%\s*thickness achieved",
    # Other phrasing we've seen in practice:
    r"thickness\s*([0-9.]+)\s*%\s*achieved",
    r"achieved\s*thickness\s*[:=]\s*([0-9.]+)\s*%",
    # snappyHexMesh standard output table format:
    r"wall_aorta\s+\d+\s+[\d.]+\s+[\d.eE-]+\s+([\d.]+)",
]

PRUNE_HINTS = [
    "maxFaceThicknessRatio", "maxThicknessToMedialRatio",
    "minMedianAxisAngle", "featureAngle", "illegal faces",
    "Removing extrusion", "Will not extrude", "Deleting layer"
]

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")

def parse_snappy_dict(dict_path: Path):
    txt = read_text(dict_path)
    # Basic fields
    N = int(re.search(SNAPPY_NAMES["nSurfaceLayers"], txt).group(1)) if re.search(SNAPPY_NAMES["nSurfaceLayers"], txt) else None
    ER = float(re.search(SNAPPY_NAMES["expansionRatio"], txt).group(1).strip()) if re.search(SNAPPY_NAMES["expansionRatio"], txt) else None
    rel_m = re.search(SNAPPY_NAMES["relativeSizes"], txt)
    relative = (rel_m and rel_m.group(1).lower() == "true")

    # Optional sizing values (not needed for N_eff, but nice to echo)
    t1_rel = tT_rel = t1_abs = T_abs = None
    if relative:
        m1 = re.search(SNAPPY_NAMES["firstLayerThickness_rel"], txt)
        mT = re.search(SNAPPY_NAMES["finalLayerThickness_rel"], txt)
        if m1: t1_rel = float(m1.group(1))
        if mT: tT_rel = float(mT.group(1))
    else:
        m1 = re.search(SNAPPY_NAMES["firstLayerThickness_abs"], txt)
        if m1: t1_abs = float(m1.group(1))
        # T_abs is not printed by SHM; computing it needs N & ER & t1. We'll compute later if needed.

    return {
        "N": N, "ER": ER, "relative": relative,
        "t1_rel": t1_rel, "T_rel": tT_rel, "t1_abs": t1_abs
    }

def scrape_thickness_fraction(log_path: Path):
    txt = read_text(log_path)
    for pat in LOG_PATTERNS:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            pct = float(m.group(1))
            return pct / 100.0
    return None

def count_prune_hints(log_path: Path):
    txt = read_text(log_path)
    hits = {k: len(re.findall(k, txt, flags=re.IGNORECASE)) for k in PRUNE_HINTS}
    total = sum(hits.values())
    return total, hits

def effective_layers(N, ER, thickness_fraction):
    """
    Geometric stack: t_k = t1 * ER^(k-1), k=1..N
    T_target/t1 = (ER^N - 1)/(ER - 1)
    N_eff = log(1 + thickness_fraction * (ER^N - 1)) / log(ER)
    (t1 cancels — works for absolute or relative sizing)
    """
    if N is None or ER is None or thickness_fraction is None:
        return None
    if ER <= 1.0:
        return thickness_fraction * N  # degenerate case; treat as linear
    return math.log(1.0 + thickness_fraction * (ER**N - 1.0)) / math.log(ER)

def diagnose(thickness_fraction, N_eff, prune_total):
    if thickness_fraction is None:
        return "unknown", "Could not find thickness % in the log. Ensure your pipeline writes a 'thickness achieved' line."
    # Heuristics
    if thickness_fraction < 0.15:
        label = "not-added-or-immediately-abandoned"
        why = "Very low thickness; likely blocked at t1 or abandoned early by strict quality."
    elif thickness_fraction < 0.60:
        label = "thin-but-present (added-then-pruned)"
        why = "Some growth occurred but much was pruned by quality gates."
    else:
        label = "healthy-growth"
        why = "Most of the requested thickness was realized."

    if prune_total > 0 and label != "healthy-growth":
        why += " Log shows significant pruning/illegal-face hints."

    # Refine by N_eff if available
    if N_eff is not None:
        if N_eff < 1.0 and label != "not-added-or-immediately-abandoned":
            label = "barely-one-layer"
            why += " Effective layers < 1."
        elif N_eff < 3.0 and label == "healthy-growth":
            label = "moderate-growth"
            why += " Effective layers still under 3."

    return label, why

def main():
    ap = argparse.ArgumentParser(description="snappyHexMesh layer log diagnostics")
    ap.add_argument("--snappy", required=True, help="Path to system/snappyHexMeshDict (the one used for layers)")
    ap.add_argument("--log", required=True, help="Path to layers log (e.g., log.snappy.layers)")
    args = ap.parse_args()

    d = parse_snappy_dict(Path(args.snappy))
    frac = scrape_thickness_fraction(Path(args.log))
    prune_total, prune_hits = count_prune_hints(Path(args.log))
    n_eff = effective_layers(d["N"], d["ER"], frac)

    label, why = diagnose(frac, n_eff, prune_total)

    print("=== Layer Diagnostics ===")
    print(f"Mode: {'relative' if d['relative'] else 'absolute'} | N={d['N']} | ER={d['ER']}")
    if d["relative"]:
        print(f"t1_rel={d['t1_rel']} | T_rel={d['T_rel']}")
    else:
        print(f"t1_abs={d['t1_abs']}")
    if frac is not None:
        print(f"Thickness% (from log): {frac*100:.1f}%")
    else:
        print("Thickness% (from log): <not found>")
    if n_eff is not None:
        print(f"Effective layers N_eff: {n_eff:.2f}")
    else:
        print("Effective layers N_eff: <n/a>")
    print(f"Pruning/illegal-face hints: {prune_total} (detail: { {k:v for k,v in prune_hits.items() if v} })")
    print(f"Diagnosis: {label} — {why}")

if __name__ == "__main__":
    main()