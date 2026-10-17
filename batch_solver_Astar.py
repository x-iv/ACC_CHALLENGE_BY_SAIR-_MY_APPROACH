"""
Batch A* solver: parse ac.jsonl, group by relator length, solve easy problems.
Run from: competition/problems/

Writes:
  - submission.txt (accepted-format lines, appended incrementally)
  - batch_log.txt  (per-problem status)
"""

import heapq
import json
import os
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
            })
    return problems

# ============================================================
# HEURISTIC (same one that solved ac-04359)
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
# A*
# ============================================================
def astar(initial, max_expansions=3_000_000, time_budget=60):
    """Short time budget — we want to grind through many easy problems."""
    t0 = time.time()
    counter = 0
    open_heap = [(h(initial), counter, 0, initial, [])]
    visited = {initial: 0}

    expansions = 0
    while open_heap:
        if time.time() - t0 > time_budget:
            return None
        if expansions > max_expansions:
            return None

        f, _, g, state, path = heapq.heappop(open_heap)
        expansions += 1

        if state == TARGET:
            return path
        if g > visited.get(state, float('inf')):
            continue

        for m in range(14):
            ns = apply_move(state, m)
            ng = g + 1
            if ng < visited.get(ns, float('inf')):
                visited[ns] = ng
                counter += 1
                heapq.heappush(open_heap, (ng + h(ns), counter, ng, ns, path + [m]))
    return None

# ============================================================
# WORKER
# ============================================================
def solve_task(args):
    cid, initial, time_budget = args
    try:
        moves = astar(initial, time_budget=time_budget)
    except Exception as e:
        return cid, None, str(e)
    return cid, moves, None

# ============================================================
# MAIN
# ============================================================
def main():
    here = Path(__file__).parent
    ac_jsonl = here / "ac.jsonl"
    submission_path = here / "submission.txt"
    log_path = here / "batch_log.txt"

    # -------- Load and sort by difficulty --------
    print("Loading problems...")
    all_problems = load_all(ac_jsonl)
    for p in all_problems:
        p["total_len"] = len(p["initial"][0]) + len(p["initial"][1])
    all_problems.sort(key=lambda p: p["total_len"])

    print(f"Total problems: {len(all_problems)}")
    print("Difficulty distribution (by total relator length):")
    for lower, upper in [(2,6),(7,10),(11,14),(15,20),(21,30),(31,100)]:
        n = sum(1 for p in all_problems if lower <= p["total_len"] <= upper)
        print(f"  length {lower:3d}-{upper:3d}: {n:5d} problems")
    print()

    # -------- Config --------
    # Adjust these to control the batch
    max_total_len = 10          # only attempt problems with total length ≤ this
    per_problem_timeout = 30    # seconds per problem
    n_workers = max(1, os.cpu_count() - 1)

    candidates = [p for p in all_problems if p["total_len"] <= max_total_len]
    print(f"Selected {len(candidates)} problems with total length ≤ {max_total_len}")
    print(f"Using {n_workers} workers, {per_problem_timeout}s budget per problem\n")

    tasks = [(p["cid"], p["initial"], per_problem_timeout) for p in candidates]

    # Open both files
    with open(submission_path, "w", encoding="utf-8") as sub, \
         open(log_path, "w", encoding="utf-8") as log:

        solved = 0
        failed = 0
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(solve_task, t): t[0] for t in tasks}

            for i, fut in enumerate(as_completed(futures), 1):
                cid, moves, err = fut.result()

                if err:
                    log.write(f"ERROR  {cid}: {err}\n")
                    failed += 1
                elif moves is not None:
                    sub.write(f"{cid}: [{', '.join(str(m) for m in moves)}]\n")
                    sub.flush()
                    log.write(f"SOLVED {cid}: len={len(moves)}\n")
                    solved += 1
                else:
                    log.write(f"FAILED {cid}\n")
                    failed += 1

                log.flush()

                if i % 20 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[{i}/{len(tasks)}] solved={solved} failed={failed} "
                          f"elapsed={elapsed:.0f}s")

    print(f"\nDone in {time.time()-t0:.0f}s.")
    print(f"Solved: {solved} / {len(candidates)}")
    print(f"Submission: {submission_path}")
    print(f"Log:        {log_path}")

if __name__ == "__main__":
    main()