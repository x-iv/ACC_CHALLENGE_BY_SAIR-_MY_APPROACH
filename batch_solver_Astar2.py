"""
Batch A* solver — 3-tier strategy with incremental submission.
Run from: competition/problems/
"""

import heapq
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- Core semantics (verifier/core.py) ----------
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

# ---------- Parser ----------
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

def load_already_solved(submission_path):
    """Return set of challenge IDs already in submission.txt."""
    done = set()
    if not submission_path.exists():
        return done
    with open(submission_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: ac-XXXXX: [moves]
            cid = line.split(":", 1)[0].strip()
            done.add(cid)
    return done

# ---------- Heuristic ----------
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

# ---------- A* ----------
def astar(initial, max_expansions=5_000_000, time_budget=60):
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

def solve_task(args):
    cid, initial, time_budget = args
    try:
        moves = astar(initial, time_budget=time_budget)
    except Exception as e:
        return cid, None, str(e)
    return cid, moves, None

# ---------- Tier runner ----------
def run_tier(problems, submission_path, log_path, label,
             time_budget, n_workers, already):
    """Solve a batch of problems, appending successes to submission.txt."""
    todo = [p for p in problems if p["cid"] not in already]
    if not todo:
        print(f"[{label}] nothing to do.")
        return 0

    print(f"\n=== {label}: {len(todo)} problems, {time_budget}s budget each ===")
    tasks = [(p["cid"], p["initial"], time_budget) for p in todo]

    solved_this_tier = 0
    failed_this_tier = 0
    t0 = time.time()

    with open(submission_path, "a", encoding="utf-8") as sub, \
         open(log_path, "a", encoding="utf-8") as log:

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(solve_task, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                cid, moves, err = fut.result()
                if err:
                    log.write(f"ERROR  {cid}: {err}\n"); failed_this_tier += 1
                elif moves is not None:
                    sub.write(f"{cid}: [{', '.join(str(m) for m in moves)}]\n")
                    sub.flush()
                    log.write(f"SOLVED {cid}: len={len(moves)}\n")
                    solved_this_tier += 1
                    already.add(cid)
                else:
                    log.write(f"FAILED {cid}\n"); failed_this_tier += 1
                log.flush()

                if i % 25 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  [{i}/{len(tasks)}] solved={solved_this_tier} "
                          f"failed={failed_this_tier} elapsed={el:.0f}s")

    print(f"[{label}] done: solved={solved_this_tier}, "
          f"failed={failed_this_tier}, total_time={time.time()-t0:.0f}s")
    return solved_this_tier

# ---------- Main ----------
def main():
    here = Path(__file__).parent
    ac_jsonl = here / "ac.jsonl"
    submission_path = here / "submission.txt"
    log_path = here / "batch_log.txt"

    print("Loading problems...")
    problems = load_all(ac_jsonl)
    for p in problems:
        p["total_len"] = len(p["initial"][0]) + len(p["initial"][1])
    problems.sort(key=lambda p: p["total_len"])
    print(f"Loaded {len(problems)} problems.")

    already = load_already_solved(submission_path)
    print(f"Already solved (from submission.txt): {len(already)}\n")

    n_workers = max(1, os.cpu_count() - 1)
    print(f"Using {n_workers} workers.\n")

    # ---- Tier configuration ----
    # (label, max_total_len, per_problem_timeout_seconds)
    tiers = [
        ("tier1_len<=14",   14, 20),
        ("tier2_len<=20",   20, 60),
        ("tier3_len<=26",   26, 120),
        ("tier4_len<=32",   32, 240),
    ]

    for label, max_len, timeout in tiers:
        batch = [p for p in problems if p["total_len"] <= max_len]
        run_tier(batch, submission_path, log_path, label, timeout, n_workers, already)
        print()

    # ---- Summary ----
    total_solved = len(already)
    print(f"\nTotal solved so far: {total_solved}")
    print(f"Submission: {submission_path}")
    print(f"Log:        {log_path}")
    print("\nVerify with:")
    print(f"  set PYTHONPATH=..\\tools")
    print(f"  python -m verifier --manifest ..\\tools\\verifier\\data\\manifest.json "
          f"--submission {submission_path.name} --pretty")

if __name__ == "__main__":
    main()