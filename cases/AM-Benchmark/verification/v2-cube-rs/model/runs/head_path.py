"""筛选用:把路径文件截断到前 N 行(保留表头)。"""
import csv
import sys

path, n = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(path)))
out = rows[:n]
w = csv.DictWriter(open(path, "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(out)
print(f"path truncated {len(rows)} -> {len(out)} rows")
