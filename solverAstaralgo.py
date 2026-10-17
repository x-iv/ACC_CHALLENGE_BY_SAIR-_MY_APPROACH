"""
A* solver for ac-00003.
Run from: competition/problems/
"""

import heapq
import time

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

cid = "ac-00003"
initial = (
    (-2,-2,-2,-2,-1,2,2,2,1),        # y⁻⁴ x⁻¹ y³ x
    (-2,-2,-2,-2,-2,1,-2,-2,-1,-1),  # y⁻⁵ x y⁻² x⁻²
)

# Stronger heuristic: total length + shape-aware bonus
def h(state):
    r0, r1 = state
    L = len(r0) + len(r1)
    # Heavy bonus for reaching the goal directly
    if state == TARGET:
        return 0
    # If r0 is just x (or x⁻¹) we're very close
    bonus = 0
    if r0 in [(1,), (-1,)]:
        bonus -= 2
    if r1 in [(2,), (-2,)]:
        bonus -= 2
    # If either relator has many y's or x's that can cancel, small bonus
    return L + bonus

def astar(initial, max_expansions=20_000_000, time_budget=900):
    t0 = time.time()
    counter = 0
    open_heap = [(h(initial), counter, 0, initial, [])]
    visited = {initial: 0}

    expansions = 0
    last_log = time.time()
    while open_heap:
        if time.time() - t0 > time_budget:
            print(f"  time budget exceeded")
            return None
        if expansions > max_expansions:
            print(f"  expansion budget exceeded")
            return None

        f, _, g, state, path = heapq.heappop(open_heap)
        expansions += 1

        if time.time() - last_log > 15:
            last_log = time.time()
            print(f"  [progress] expanded={expansions:,}, "
                  f"g={g}, h={h(state)}, queue={len(open_heap):,}, "
                  f"visited={len(visited):,}, "
                  f"elapsed={time.time()-t0:.0f}s")

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

print(f"=== {cid} ===")
print(f"  r0 = {initial[0]}  len={len(initial[0])}")
print(f"  r1 = {initial[1]}  len={len(initial[1])}")
print(f"  target = {TARGET}\n")
print("Searching...")

t0 = time.time()
moves = astar(initial)
elapsed = time.time() - t0

if moves is None:
    print(f"\nNOT SOLVED in {elapsed:.1f}s")
else:
    print(f"\nSOLVED in {len(moves)} moves, {elapsed:.1f}s")
    print(f"moves = {moves}")
    state = initial
    for m in moves:
        state = apply_move(state, m)
    print(f"final = {state}")
    print(f"match = {state == TARGET}")
    print(f"\n>>> SUBMISSION LINE:")
    print(f"{cid}: [{', '.join(str(m) for m in moves)}]")