"""Subsample the scan-path CSV by a stride, preserving energy and layer info.

dt in the solver is derived from the time column of consecutive kept rows, so
a stride of S automatically yields dt = S * dt_original while laser power is
unchanged: deposited energy per unit path length is conserved. Layer
transitions survive stride subsampling (recoat insertion triggers on the
layer id change between consecutive states); the final row is always kept.

Usage: python subsample_path.py <src.csv> <dst.csv> <stride>
"""
import csv
import sys


def main():
    src, dst, stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if stride < 1:
        raise SystemExit("stride must be >= 1")
    with open(src, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        fields = reader.fieldnames
    kept = rows[::stride]
    if rows and (len(rows) - 1) % stride != 0:
        kept.append(rows[-1])
    with open(dst, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    layers = sorted({int(r["layer"]) for r in kept})
    print(f"subsample stride={stride}: {len(rows)} -> {len(kept)} rows, "
          f"layers {layers[0]}..{layers[-1]}")


if __name__ == "__main__":
    main()
