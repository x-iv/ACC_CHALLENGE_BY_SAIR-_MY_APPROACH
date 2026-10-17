"""
Iterative Deepening DFS solver for specific AC problems.
Run from: competition/problems/
"""

import sys
from pathlib import Path

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

# ---------- Iterative Deepening DFS ----------
def iddfs(initial, max_depth, counter):
    """
    Iterative deepening DFS. Tries depth 1, 2, 3, ..., max_depth.
    Returns list of moves, or None.
    """
    for depth in range(1, max_depth + 1):
        found = _dfs(initial, [], depth, set(), counter)
        if found is not None:
            return found
    return None

def _dfs(state, path, depth_left, on_path, counter):
    """Recursive DFS with a depth limit."""
    counter[0] += 1
    if state == TARGET:
        return list(path)
    if depth_left == 0:
        return None

    for m in range(14):
        ns = apply_move(state, m)
        if ns in on_path:
            continue  # avoid cycles on the current path
        on_path.add(ns)
        path.append(m)
        result = _dfs(ns, path, depth_left - 1, on_path, counter)
        if result is not None:
            return result
        path.pop()
        on_path.remove(ns)

    return None

# ---------- Main ----------
def main():
    # The three problems you want to solve
    problems = {
        "ac-09980": (
            (-2, -2, 1, 1, 1),                          # r0 = y^-1 y^-1 x x x
            (-2, -1, -2, -1, -1, 2, -1, -1, 2, 1, 2, 1, 1)  # r1 = y^-1 x^-1 y^-1 x^-1 x^-1 y x^-1 x^-1 y x y x x
        ),
        "ac-09650": (
            (-2, -2, 1, 2, 1),                          # r0 = y^-1 y^-1 x y x
            (-2, -2, -1, -2, 1, 1, 2, 2, 1, 1, -2, 1, 1)    # r1 = y^-1 y^-1 x^-1 y^-1 x x y y x x y^-1 x x
        ),
    }

    max_depth = 15  # try up to 15 moves deep

    for cid, initial in problems.items():
        print(f"\n=== {cid} ===")
        print(f"  r0 = {initial[0]}")
        print(f"  r1 = {initial[1]}")
        counter = [0]
        moves = iddfs(initial, max_depth, counter)
        if moves is None:
            print(f"  NOT SOLVED within depth {max_depth}  ({counter[0]:,} nodes explored)")
        else:
            print(f"  SOLVED in {len(moves)} moves  ({counter[0]:,} nodes explored)")
            print(f"  moves = {moves}")
            print(f"\n  Submission line:")
            print(f"    {cid}: [{', '.join(str(m) for m in moves)}]")

            # Verify
            state = initial
            for m in moves:
                state = apply_move(state, m)
            print(f"\n  Verification: final={state}, target={TARGET}, match={state == TARGET}")

if __name__ == "__main__":
    main()