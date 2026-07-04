import json
import sys


def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["stage"]] = r
    return out


def main(ours_path, ref_path, ours_ids, ref_ids):
    a = open(ours_ids).read().strip().split(",")
    b = open(ref_ids).read().strip().split(",")
    if a == b:
        print(f"PROMPT IDS: identical ({len(a)} tokens)")
    else:
        print(f"PROMPT IDS DIFFER: ours={len(a)} ref={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  first diff at pos {i}: ours={x} ref={y}")
                print(f"  ours[{max(0,i-2)}:{i+3}] = {a[max(0,i-2):i+3]}")
                print(f"  ref [{max(0,i-2)}:{i+3}] = {b[max(0,i-2):i+3]}")
                break
        if len(a) != len(b):
            print("  (length mismatch)")

    ours, ref = load(ours_path), load(ref_path)
    stages = ["embeddings"] + [f"layer{i}" for i in range(40)] + ["final_norm"]
    print(f"{'stage':<12}{'ours_norm':>12}{'ref_norm':>12}{'norm_ratio':>12}{'max_head_diff':>15}")
    first_bad = None
    for s in stages:
        o, r = ours.get(s), ref.get(s)
        if not o or not r:
            print(f"{s:<12} MISSING {'ours' if not o else 'ref'}")
            continue
        ratio = o["norm"] / r["norm"] if r["norm"] else float("inf")
        hd = max(abs(x - y) for x, y in zip(o["head"], r["head"]))
        rel = hd / (max(abs(v) for v in r["head"]) + 1e-9)
        flag = ""
        if (abs(ratio - 1) > 0.05 or rel > 0.2) and first_bad is None:
            first_bad = s
            flag = "  <-- FIRST DIVERGENCE"
        print(f"{s:<12}{o['norm']:>12.4f}{r['norm']:>12.4f}{ratio:>12.4f}{hd:>15.5f}{flag}")
    print("\nfirst clearly diverging stage:", first_bad or "none (within tolerance)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
