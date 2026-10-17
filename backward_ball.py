"""
Precompute the backward ball from TARGET up to depth K.
Saves to ball_k{K}.pkl. Run ONCE.
Run from: competition/problems/
"""

import pickle
import time
from collections import deque
from pathlib import Path

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
K = 5   # depth of the backward ball

print(f"Building backward ball from TARGET, K={K} ...")
t0 = time.time()

ball = {TARGET: []}
frontier = deque([(TARGET, [])])
for depth in range(K):
    next_frontier = deque()
    while frontier:
        state, path = frontier.popleft()
        for m in range(14):
            ns = apply_move(state, m)
            if ns not in ball:
                ball[ns] = [m] + path   # path from ns to TARGET
                next_frontier.append((ns, ball[ns]))
    frontier = next_frontier
    print(f"  depth {depth+1}: {len(ball):,} states total")

out = Path(f"ball_k{K}.pkl")
with open(out, "wb") as f:
    pickle.dump(ball, f)

print(f"\nWrote {out} with {len(ball):,} states in {time.time()-t0:.1f}s")