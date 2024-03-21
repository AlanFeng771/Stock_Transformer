from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
import matplotlib.pyplot as plt
pred = np.load(r'results\mamba_result\pred.npy').squeeze(axis=1)
label = np.load(r'results\mamba_result\true.npy').squeeze(axis=1)
pred_high = pred[:, 2]
label_high = label[:, 2]
x_range = np.arange(383)
plt.plot(x_range, pred_high, label='pred')
plt.plot(x_range, label_high, label='label')
plt.legend()
plt.show()

