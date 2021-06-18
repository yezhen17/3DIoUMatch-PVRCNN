import numpy as np
import sys

ratio = float(sys.argv[1])
num = int(sys.argv[2])

with open("train.txt", "r") as f:
    lines = f.read().strip().split('\n')
    inds = np.random.choice(len(lines), int(len(lines)*ratio), replace=False)
    newlines= []
    for i in inds:
        newlines.append(f'{lines[i]} {i}')

with open("train_%.2f_%d.txt" % (ratio, num), "w") as fw:
    fw.write('\n'.join(newlines))

