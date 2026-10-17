"""
Bidirectional BFS for the same length-bucketed problems.
Fresh implementation, separate from A*. 60s per problem.
Writes submission_bfs.txt and bfs_log.jsonl.
Run from: competition/problems/
"""

import json
import time
from collections import deque
from pathlib import Path

# ============================================================
# CORE SEMANTICS
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
# BIDIRECTIONAL BFS
# ============================================================
def bidir_bfs(initial, time_budget=60, max_states=2_000_000):
    """
    Returns (moves, trace, states_expanded) or (None, None, n).
    """
    t0 = time.time()
    if initial == TARGET:
        return [], [], 0

    fwd = deque([(initial, [])])
    bwd = deque([(TARGET, [])])
    fwd_seen = {initial: []}
    bwd_seen = {TARGET: []}
    expanded = 0

    def build_trace(path):
        """Given the move list, produce a trace dictionary per step."""
        trace = []
        cur = initial
        for i, m in enumerate(path):
            nxt = apply_move(cur, m)
            trace.append({
                "step": i,
                "move": m,
                "before": [list(cur[0]), "|", list(cur[1])],
                "after":  [list(nxt[0]), "|", list(nxt[1])],
            })
            cur = nxt
        return trace

    def expand(frontier, seen_self, seen_other, direction):
        nonlocal expanded
        for _ in range(len(frontier)):
            if time.time() - t0 > time_budget:
                return None
            state, path = frontier.popleft()
            expanded += 1
            for m in range(14):
                ns = apply_move(state, m)
                if ns in seen_other:
                    if direction == "fwd":
                        full = path + [m] + seen_other[ns]
                    else:
                        full = seen_other[ns] + [m] + path
                    return full
                if ns not in seen_self:
                    seen_self[ns] = path + [m]
                    frontier.append((ns, path + [m]))
        return None

    while fwd or bwd:
        if time.time() - t0 > time_budget:
            return None, None, expanded
        if len(fwd_seen) + len(bwd_seen) > max_states:
            return None, None, expanded

        # Expand smaller frontier
        if len(fwd) <= len(bwd):
            result = expand(fwd, fwd_seen, bwd_seen, "fwd")
        else:
            result = expand(bwd, bwd_seen, fwd_seen, "bwd")

        if result is not None:
            return result, build_trace(result), expanded

    return None, None, expanded

# ============================================================
# MAIN
# ============================================================
def main():
    here = Path(__file__).parent
    ac_jsonl = here / "ac.jsonl"
    submission_path = here / "submission_bfs.txt"
    log_path = here / "bfs_log.jsonl"

    # Same set of problems as the A* run
    max_total_len = 14
    time_budget = 60  # strictly 60s per problem

    print(f"Loading problems with total length <= {max_total_len} ...")
    all_problems = load_all(ac_jsonl)
    problems = [p for p in all_problems
                if len(p["initial"][0]) + len(p["initial"][1]) <= max_total_len]
    problems.sort(key=lambda p: len(p["initial"][0]) + len(p["initial"][1]))

    print(f"Found {len(problems)} problems.")
    print(f"Time budget: {time_budget}s per problem.\n")

    solved = 0
    with open(submission_path, "w", encoding="utf-8") as sub, \
         open(log_path, "w", encoding="utf-8") as log:

        for i, p in enumerate(problems, 1):
            cid = p["cid"]
            initial = p["initial"]
            total_len = len(initial[0]) + len(initial[1])
            print(f"[{i}/{len(problems)}] {cid} (len={total_len}) ", end="", flush=True)

            t0 = time.time()
            moves, trace, expanded = bidir_bfs(initial, time_budget=time_budget)
            elapsed = time.time() - t0

            if moves is None:
                print(f"FAILED ({elapsed:.1f}s, {expanded:,} states)")
            else:
                print(f"SOLVED ({len(moves)} moves, {elapsed:.1f}s, {expanded:,} states)")
                sub.write(f"{cid}: [{', '.join(str(m) for m in moves)}]\n")
                sub.flush()
                solved += 1

                log.write(json.dumps({
                    "challenge_id": cid,
                    "algorithm": "bidir_bfs",
                    "length": len(moves),
                    "moves": moves,
                    "initial": [list(initial[0]), list(initial[1])],
                    "trace": trace,
                    "expanded": expanded,
                    "solved_at": elapsed,
                }) + "\n")
                log.flush()

    print(f"\nSolved {solved}/{len(problems)}.")
    print(f"Submission: {submission_path}")
    print(f"Log:        {log_path}")

if __name__ == "__main__":
    main()