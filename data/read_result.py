import numpy as np

file_name = 'result_isic.txt'

with open(file_name, 'r') as f:
    dat = f.readlines()

dat = list(map(lambda x: x.strip().split(' & ')[1:], dat))

dat = np.asarray(dat, dtype=np.float32)

temp = dat.mean(0).round(2)

print(' & '.join(map(str, temp)))