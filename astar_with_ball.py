"""
A* forward search that uses the precomputed backward ball.
Whenever a forward state is in the ball, we're done.
Run from: competition/problems/
"""

import heapq
import json
import pickle
import time
from pathlib import Path

# ---- core semantics (same as before) ----
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

# ---- heuristic ----
def h(state):
    r0, r1 = state
    L = len(r0) + len(r1)
    if state == TARGET: return 0
    bonus = 0
    if r0 in [(1,), (-1,)]: bonus -= 2
    if r1 in [(2,), (-2,)]: bonus -= 2
    return L + bonus

# ---- A* with ball ----
def astar_with_ball(initial, ball, time_budget=60, max_expansions=10_000_000):
    t0 = time.time()
    if initial == TARGET:
        return []
    if initial in ball:
        return list(ball[initial])

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
        if g > visited.get(state, float('inf')):
            continue

        for m in range(14):
            ns = apply_move(state, m)
            # HIT THE BALL!
            if ns in ball:
                return path + [m] + list(ball[ns])
            ng = g + 1
            if ng < visited.get(ns, float('inf')):
                visited[ns] = ng
                counter += 1
                heapq.heappush(open_heap, (ng + h(ns), counter, ng, ns, path + [m]))

    return None

# ---- parser ----
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
        if not chunk: continue
        w = encode(chunk.replace("= 1", "").strip())
        relators.append(tuple(CHAR_TO_INT[c] for c in w))
    return relators

def load_all(path):
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            rels = parse_description(obj["description"])
            if len(rels) < 2: continue
            problems.append({"cid": obj["challenge_id"], "initial": (rels[0], rels[1])})
    return problems

# ---- main ----
def main():
    here = Path(__file__).parent
    with open(here / "ball_k5.pkl", "rb") as f:
        ball = pickle.load(f)
    print(f"Loaded ball with {len(ball):,} states")

    all_problems = load_all(here / "ac.jsonl")
    problems = [p for p in all_problems
                if len(p["initial"][0]) + len(p["initial"][1]) <= 14]

    submission = here / "submission_ball.txt"
    solved = 0
    with open(submission, "w") as sub:
        for i, p in enumerate(problems, 1):
            cid, initial = p["cid"], p["initial"]
            print(f"[{i}/{len(problems)}] {cid} ", end="", flush=True)
            t0 = time.time()
            moves = astar_with_ball(initial, ball, time_budget=60)
            elapsed = time.time() - t0
            if moves is None:
                print(f"FAILED ({elapsed:.1f}s)")
            else:
                print(f"SOLVED ({len(moves)} moves, {elapsed:.1f}s)")
                sub.write(f"{cid}: [{', '.join(str(m) for m in moves)}]\n")
                sub.flush()
                solved += 1
    print(f"\nSolved {solved}/{len(problems)}")

if __name__ == "__main__":
    main()