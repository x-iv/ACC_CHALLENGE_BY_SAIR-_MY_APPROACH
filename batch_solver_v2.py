"""
Batch A* solver with full per-move tracing and live terminal output.

Writes:
  submission.txt     - strict format for SAIR upload
  analysis_log.jsonl - rich log for later analysis (one JSON per solved problem)
  batch_log.txt      - high-level run log

Run from: competition/problems/
"""

import heapq
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# CORE SEMANTICS (verifier/core.py)
# ============================================================
def free_reduce(word):
    out = []
    for a in word:
        if out and out[-1] == -a:
            out.pop()
        else:
            out.append(a)
    return tuple(out)

def invert(word):
    return tuple(-a for a in reversed(word))

GENS = (1, -1, 2, -2)

def apply_move(state, m):
    r0, r1 = state
    if m == 0:  return (invert(r0), r1)
    if m == 1:  return (r0, invert(r1))
    if m == 2:  return (free_reduce(r0 + r1), r1)
    if m == 3:  return (free_reduce(r0 + invert(r1)), r1)
    if m == 4:  return (r0, free_reduce(r1 + r0))
    if m == 5:  return (r0, free_reduce(r1 + invert(r0)))
    if 6 <= m <= 9:
        g = GENS[m - 6]
        return (free_reduce((g,) + r0 + (-g,)), r1)
    if 10 <= m <= 13:
        g = GENS[m - 10]
        return (r0, free_reduce((g,) + r1 + (-g,)))
    raise ValueError("bad move id: %r" % (m,))

TARGET = ((1,), (2,))

# ============================================================
# PARSER
# ============================================================
CHAR_TO_INT = {"x": 1, "X": -1, "y": 2, "Y": -2}

def encode(word):
    return (word.replace("x^-1", "X")
                .replace("y^-1", "Y")
                .replace(" ", ""))

def parse_description(desc):
    s = desc.strip().replace("Presentation:", "").strip().rstrip(".").strip()
    _, rels_part = s.strip("<>").split("|", 1)
    relators = []
    for chunk in rels_part.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        w = encode(chunk.replace("= 1", "").strip())
        relators.append(tuple(CHAR_TO_INT[c] for c in w))
    return relators

def load_all(path):
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rels = parse_description(obj["description"])
            if len(rels) < 2:
                continue
            problems.append({
                "cid": obj["challenge_id"],
                "initial": (rels[0], rels[1]),
                "total_len": len(rels[0]) + len(rels[1]),
            })
    return problems

# ============================================================
# HEURISTIC
# ============================================================
def h(state):
    r0, r1 = state
    L = len(r0) + len(r1)
    if state == TARGET:
        return 0
    bonus = 0
    if r0 in [(1,), (-1,)]:
        bonus -= 2
    if r1 in [(2,), (-2,)]:
        bonus -= 2
    return L + bonus

# ============================================================
# A* WITH TRACE
# ============================================================
def astar_with_trace(initial, max_expansions=5_000_000, time_budget=60):
    """
    A* that returns (moves, trace).
    trace is a list of dicts describing each step:
      {
        "step": i,
        "move": m,
        "state_before": (r0, r1),
        "state_after": (r0, r1),
        "g": path length so far,
        "h": heuristic of new state,
      }
    """
    t0 = time.time()
    if initial == TARGET:
        return [], []

    counter = 0
    open_heap = [(h(initial), counter, 0, initial, [])]
    visited = {initial: 0}

    expansions = 0
    while open_heap:
        if time.time() - t0 > time_budget:
            return None, None
        if expansions > max_expansions:
            return None, None

        f, _, g, state, path = heapq.heappop(open_heap)
        expansions += 1

        if state == TARGET:
            # Reconstruct trace by replaying
            trace = []
            s = initial
            for i, m in enumerate(path):
                ns = apply_move(s, m)
                trace.append({
                    "step": i,
                    "move": m,
                    "state_before": list(s[0]) + ["|"] + list(s[1]),
                    "state_after": list(ns[0]) + ["|"] + list(ns[1]),
                    "g": i + 1,
                    "h": h(ns),
                })
                s = ns
            return path, trace

        if g > visited.get(state, float('inf')):
            continue

        for m in range(14):
            ns = apply_move(state, m)
            ng = g + 1
            if ng < visited.get(ns, float('inf')):
                visited[ns] = ng
                counter += 1
                heapq.heappush(open_heap, (ng + h(ns), counter, ng, ns, path + [m]))

    return None, None

# ============================================================
# WORKER
# ============================================================
def solve_task(args):
    cid, initial, time_budget = args
    try:
        moves, trace = astar_with_trace(initial, time_budget=time_budget)
    except Exception as e:
        return cid, None, None, str(e)
    return cid, moves, trace, None

# ============================================================
# FORMATTING HELPERS
# ============================================================
def format_moves_line(cid, moves):
    """Strict format the verifier accepts. No comments, no trailing comma."""
    return f"{cid}: [{', '.join(str(m) for m in moves)}]"

# ============================================================
# MAIN
# ============================================================
def main():
    here = Path(__file__).parent
    ac_jsonl = here / "ac.jsonl"
    submission_path = here / "submission.txt"
    analysis_path = here / "analysis_log.jsonl"
    batch_log_path = here / "batch_log.txt"

    # -------- Load --------
    print("Loading problems...")
    all_problems = load_all(ac_jsonl)
    all_problems.sort(key=lambda p: p["total_len"])
    print(f"Total problems: {len(all_problems)}\n")

    # -------- Config --------
    MAX_TOTAL_LEN = 14
    PER_PROBLEM_TIMEOUT = 120   # seconds
    N_WORKERS = max(1, os.cpu_count() - 1)
    # ------------------------

    candidates = [p for p in all_problems if p["total_len"] <= MAX_TOTAL_LEN]
    print(f"Selected {len(candidates)} problems with total length <= {MAX_TOTAL_LEN}")
    print(f"Workers: {N_WORKERS}, timeout: {PER_PROBLEM_TIMEOUT}s\n")

    tasks = [(p["cid"], p["initial"], PER_PROBLEM_TIMEOUT) for p in candidates]

    # Open all three files
    sub_file = open(submission_path, "w", encoding="utf-8")
    log_file = open(analysis_path, "w", encoding="utf-8")
    batch_log = open(batch_log_path, "w", encoding="utf-8")

    solved = 0
    failed = 0
    t0 = time.time()

    print("=" * 70)
    print("LIVE RESULTS")
    print("=" * 70)

    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(solve_task, t): t[0] for t in tasks}

            for i, fut in enumerate(as_completed(futures), 1):
                cid, moves, trace, err = fut.result()

                if err:
                    failed += 1
                    msg = f"[{i}/{len(tasks)}] {cid}: ERROR {err}"
                    print(msg)
                    batch_log.write(msg + "\n")
                    batch_log.flush()
                    continue

                if moves is None:
                    failed += 1
                    msg = f"[{i}/{len(tasks)}] {cid}: NOT SOLVED"
                    print(msg)
                    batch_log.write(msg + "\n")
                    batch_log.flush()
                    continue

                # SOLVED
                solved += 1
                line = format_moves_line(cid, moves)
                sub_file.write(line + "\n")
                sub_file.flush()

                # Rich analysis record
                record = {
                    "challenge_id": cid,
                    "length": len(moves),
                    "moves": moves,
                    "initial": [list(candidates[i-1]["initial"][0]),
                                list(candidates[i-1]["initial"][1])] if i-1 < len(candidates) else None,
                    "trace": trace,
                    "solved_at": time.time() - t0,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

                msg = f"[{i}/{len(tasks)}] {cid}: SOLVED in {len(moves)} moves | {line}"
                print(msg)
                batch_log.write(msg + "\n")
                batch_log.flush()

    finally:
        sub_file.close()
        log_file.close()
        batch_log.close()

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Done in {elapsed:.0f}s.")
    print(f"Solved:  {solved}/{len(candidates)}")
    print(f"Failed:  {failed}")
    print(f"Submission: {submission_path}")
    print(f"Analysis:   {analysis_path}")
    print(f"Batch log:  {batch_log_path}")

if __name__ == "__main__":
    main()