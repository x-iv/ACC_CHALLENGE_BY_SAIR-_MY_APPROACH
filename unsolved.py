#!/usr/bin/env python3
"""
build_unsolved_jsonl.py

Reads unsolved.txt (reference list of unsolved IDs) and the full problem
JSONL files, then writes out filtered JSONL files containing only the
unsolved problems.

Usage:
    python build_unsolved_jsonl.py \
        --unsolved unsolved.txt \
        --ac-jsonl ac.jsonl \
        --sac-jsonl sac.jsonl \
        --out-dir ./unsolved

If you only have one combined JSONL, pass it to both --ac-jsonl and --sac-jsonl,
or use --combined-jsonl.

Outputs:
    ac_unsolved.jsonl
    sac_unsolved.jsonl
    unsolved_combined.jsonl
"""

import argparse
import json
import os
import re
from typing import Set, Dict, Any, Iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_id(cid: str) -> str:
    """Strip 'ac-' / 'sac-' prefix and whitespace. Returns numeric string."""
    cid = cid.strip()
    return re.sub(r'^(sac|ac)-', '', cid, flags=re.IGNORECASE)


def load_unsolved_ids(path: str) -> Set[str]:
    """
    Load the unsolved.txt reference file.
    Returns a set of normalized numeric IDs (e.g. '00002').

    Handles lines like:
        ac-00002
        sac-00002
        00002
        ac-00002: [0, 1, 2]
    """
    ids = set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"unsolved reference file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Match "<id>" or "<id>: [...]"
            m = re.match(r'^([A-Za-z]+-\d+|\d+)\s*(?::|$)', line)
            if m:
                ids.add(normalize_id(m.group(1)))
                continue
            # Fallback: first whitespace-separated token
            tok = line.split()[0]
            if tok:
                ids.add(normalize_id(tok))

    return ids


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield each non-empty JSONL line as a dict."""
    if not os.path.exists(path):
        print(f"  [warn] {path} not found — skipping")
        return
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] {path}:{lineno} bad JSON: {e}")


def extract_challenge_id(obj: Dict[str, Any]) -> str:
    """Extract challenge_id from a problem object."""
    for key in ("challenge_id", "id", "problem_id", "name"):
        if key in obj:
            return str(obj[key])
    raise KeyError(f"no challenge_id field: {list(obj.keys())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build ac_unsolved.jsonl and sac_unsolved.jsonl from unsolved.txt."
    )
    ap.add_argument("--unsolved", default="unsolved.txt",
                    help="Reference file listing unsolved IDs (default: unsolved.txt)")
    ap.add_argument("--ac-jsonl", default="ac.jsonl",
                    help="Full AC problem set JSONL (default: ac.jsonl)")
    ap.add_argument("--sac-jsonl", default="sac.jsonl",
                    help="Full SAC problem set JSONL (default: sac.jsonl)")
    ap.add_argument("--combined-jsonl", default=None,
                    help="Use this single JSONL for both AC and SAC.")
    ap.add_argument("--out-dir", default=".",
                    help="Directory for output JSONL files (default: current dir)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ac_path  = args.combined_jsonl or args.ac_jsonl
    sac_path = args.combined_jsonl or args.sac_jsonl

    print("=" * 60)
    print("Build Unsolved JSONL")
    print("=" * 60)
    print(f"Unsolved ref: {args.unsolved}")
    print(f"AC  JSONL  : {ac_path}")
    print(f"SAC JSONL  : {sac_path}")
    print(f"Output dir : {args.out_dir}")
    print()

    # ---- Load unsolved IDs ----
    unsolved_ids = load_unsolved_ids(args.unsolved)
    print(f"Loaded {len(unsolved_ids)} unsolved IDs from {args.unsolved}")
    print()

    # ---- Output files ----
    out_ac_path   = os.path.join(args.out_dir, "ac_unsolved.jsonl")
    out_sac_path  = os.path.join(args.out_dir, "sac_unsolved.jsonl")
    out_comb_path = os.path.join(args.out_dir, "unsolved_combined.jsonl")

    f_ac   = open(out_ac_path,   'w', encoding='utf-8')
    f_sac  = open(out_sac_path,  'w', encoding='utf-8')
    f_comb = open(out_comb_path, 'w', encoding='utf-8')

    # ---- Process AC ----
    print("Processing AC problems ...")
    n_ac_total = n_ac_written = 0
    seen_ac = set()
    for obj in iter_jsonl(ac_path):
        n_ac_total += 1
        try:
            cid = extract_challenge_id(obj)
        except KeyError as e:
            print(f"  [warn] skipping AC object: {e}")
            continue
        norm = normalize_id(cid)
        if norm in seen_ac:
            continue
        seen_ac.add(norm)

        if norm in unsolved_ids:
            f_ac.write(json.dumps(obj) + "\n")
            n_ac_written += 1

    print(f"  AC total:    {n_ac_total}")
    print(f"  AC written:  {n_ac_written}")
    print()

    # ---- Process SAC ----
    print("Processing SAC problems ...")
    n_sac_total = n_sac_written = 0
    seen_sac = set()
    for obj in iter_jsonl(sac_path):
        n_sac_total += 1
        try:
            cid = extract_challenge_id(obj)
        except KeyError as e:
            print(f"  [warn] skipping SAC object: {e}")
            continue
        norm = normalize_id(cid)
        if norm in seen_sac:
            continue
        seen_sac.add(norm)

        if norm in unsolved_ids:
            f_sac.write(json.dumps(obj) + "\n")
            n_sac_written += 1

    print(f"  SAC total:   {n_sac_total}")
    print(f"  SAC written: {n_sac_written}")
    print()

    # ---- Process combined (union, deduplicated by numeric ID) ----
    print("Processing combined ...")
    n_comb_written = 0
    seen_comb = set()

    # Prefer SAC objects for the combined file; fall back to AC if SAC missing.
    # We do two passes: first SAC, then AC only if not already seen.
    for obj in iter_jsonl(sac_path):
        try:
            cid = extract_challenge_id(obj)
        except KeyError:
            continue
        norm = normalize_id(cid)
        if norm in unsolved_ids and norm not in seen_comb:
            seen_comb.add(norm)
            f_comb.write(json.dumps(obj) + "\n")
            n_comb_written += 1

    for obj in iter_jsonl(ac_path):
        try:
            cid = extract_challenge_id(obj)
        except KeyError:
            continue
        norm = normalize_id(cid)
        if norm in unsolved_ids and norm not in seen_comb:
            seen_comb.add(norm)
            f_comb.write(json.dumps(obj) + "\n")
            n_comb_written += 1

    print(f"  Combined written: {n_comb_written}")
    print()

    # ---- Close ----
    f_ac.close()
    f_sac.close()
    f_comb.close()

    # ---- Summary ----
    print("=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"Wrote {out_ac_path}   ({n_ac_written} problems)")
    print(f"Wrote {out_sac_path}  ({n_sac_written} problems)")
    print(f"Wrote {out_comb_path} ({n_comb_written} problems)")
    print()


if __name__ == "__main__":
    main()