import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_bp", required=True)
    parser.add_argument("--sim_cc", required=True)
    parser.add_argument("--sim_mf", required=True)
    parser.add_argument("--probe", default=None, help="Optional: check a specific protein ID")
    parser.add_argument("--context", type=int, default=3, help="Probe modunda etraftaki kaç satır gösterilsin")
    args = parser.parse_args()

    files = {"BP": args.sim_bp, "CC": args.sim_cc, "MF": args.sim_mf}

    anchors = set()
    candidates = set()

    for asp, fpath in files.items():
        with open(fpath) as f:
            f.readline()
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                anchors.add(cols[0])
                candidates.add(cols[1])

    all_proteins = anchors | candidates
    only_anchor    = anchors - candidates
    only_candidate = candidates - anchors
    both           = anchors & candidates

    print(f"Unique anchor                : {len(anchors)}")
    print(f"Unique candidate             : {len(candidates)}")
    print(f"Toplam unique protein        : {len(all_proteins)}")
    print(f"Sadece anchor (cand. değil)  : {len(only_anchor)}")
    print(f"Sadece candidate (anc. değil): {len(only_candidate)}")
    print(f"Hem anchor hem candidate     : {len(both)}")

    if args.probe:
        print(f"\n--- Probe: {args.probe} ---")
        for asp, fpath in files.items():
            df = pd.read_csv(fpath, sep='\t', header=0,
                             names=["anchor", "candidate", "sim", "col4", "col5"],
                             usecols=[0, 1, 2])
            as_anchor_idx    = df.index[df["anchor"] == args.probe].tolist()
            as_candidate_idx = df.index[df["candidate"] == args.probe].tolist()

            print(f"\n  [{asp}] anchor olarak {len(as_anchor_idx)} satır, candidate olarak {len(as_candidate_idx)} satır")

            if as_anchor_idx:
                first = as_anchor_idx[0]
                lo = max(0, first - args.context)
                hi = min(len(df) - 1, first + args.context)
                print(f"  İlk anchor satırı (idx={first}) ve etrafı:")
                print(df.iloc[lo:hi+1].to_string())

            if as_candidate_idx:
                first = as_candidate_idx[0]
                lo = max(0, first - args.context)
                hi = min(len(df) - 1, first + args.context)
                print(f"  İlk candidate satırı (idx={first}) ve etrafı:")
                print(df.iloc[lo:hi+1].to_string())

if __name__ == "__main__":
    main()
