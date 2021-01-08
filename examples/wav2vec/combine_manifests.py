import os
import os.path
import sys

fnames = []
entries = []
cur_root = ''
for line in sys.stdin:
    line = line.strip()
    line = line.split(None, 1)
    if len(line) == 1:
        cur_root = line[0]
    else:
        fname, extra = line
        fnames.append(os.path.join(cur_root, fname))
        entries.append(extra)

root = os.path.commonpath(fnames)
print(root)
for fname, extra in zip(fnames, entries):
    print(f"{os.path.relpath(fname, root)}\t{extra}")
