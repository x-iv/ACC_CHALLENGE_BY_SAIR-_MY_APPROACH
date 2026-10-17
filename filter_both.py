import json
from pathlib import Path

def rel_len_from_desc(desc):
    """Total length of all relators in a presentation description."""
    s = desc.replace("Presentation:", "").strip().rstrip(".").strip()
    # Everything after the pipe
    _, rest = s.strip("<>").split("|", 1)
    # Remove trailing ">."
    rest = rest.rstrip(">.")
    total = 0
    for chunk in rest.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Remove "= 1"
        chunk = chunk.replace("= 1", "").strip()
        # Count only letters (drop spaces)
        letters = chunk.replace(" ", "")
        # Count characters excluding ^ and -1 patterns
        # Simpler: use a regex-free approach
        i = 0
        while i < len(letters):
            if i + 3 < len(letters) and letters[i+1] == "^" and letters[i+2] == "-" and letters[i+3] == "1":
                total += 1  # counted once
                i += 4
            else:
                total += 1
                i += 1
    return total

def filter_file(src, dst, max_len):
    kept = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if rel_len_from_desc(obj["description"]) <= max_len:
                    fout.write(line + "\n")
                    kept += 1
            except Exception as e:
                print(f"skip: {e}")
    print(f"{src} -> {dst}: kept {kept} (len <= {max_len})")

filter_file("ac.jsonl", "ac_len14.jsonl", 14)
filter_file("stable_ac.jsonl", "sac_len14.jsonl", 14)