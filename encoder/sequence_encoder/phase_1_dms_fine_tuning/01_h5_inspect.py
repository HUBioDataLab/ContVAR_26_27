# src/scripts/01_h5_inspect.py

import os
import h5py

H5_PATH = "data/raw/embeddings_variable.h5"


def main():

    if not os.path.exists(H5_PATH):
        print("[ERROR] H5 file not found:", H5_PATH)
        return


    print("[INFO] Opening:", H5_PATH)
    f = h5py.File(H5_PATH, "r")

    keys = list(f.keys())
    print("[INFO] Number of datasets:", len(keys))


    print("\n[INFO] First 20 dataset names:")
    for k in keys[:20]:
        print("  ", k)


    print("\n[INFO] Example dataset shapes/dtypes (first 10):")
    for k in keys[:10]:
        ds = f[k]          
        print("  ", k, "shape=", ds.shape, "dtype=", ds.dtype)

    wt_count = 0
    wt_examples = []

    for k in keys:
        if k.endswith("_WT"):
            wt_count += 1
            if len(wt_examples) < 10:
                wt_examples.append(k)

    print("\n[INFO] WT count:", wt_count)
    print("[INFO] WT examples (up to 10):")
    for w in wt_examples:
        print("  ", w)

  
    label0 = 0
    label1 = 0
    other = 0

    for k in keys:
        if k.endswith("_0"):
            label0 += 1
        elif k.endswith("_1"):
            label1 += 1
        else:
            other += 1

    print("\n[INFO] name suffix counts:")
    print("  endswith _0:", label0)
    print("  endswith _1:", label1)
    print("  other      :", other)

  
    first = keys[0]
    first_shape = f[first].shape
    if len(first_shape) == 2:
        L, H = first_shape
        print("\n[INFO] First dataset shape looks like (L,H) =", (L, H))
        print("[INFO] So embedding dimension H =", H)
    else:
        print("\n[INFO] First dataset shape is not 2D:", first_shape)


    f.close()
    print("\n[DONE] Inspect finished.")


if __name__ == "__main__":
    main()