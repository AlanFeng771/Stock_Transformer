from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
import matplotlib.pyplot as plt
pred = np.load(r'results\stock01_ns_Transformer_stock_ftM_sl30_ll0_pl1_dm512_nh8_el5_dl3_df2048_fc3_ebtimeF_dtTrue_Exp_stock04_0\pred.npy').squeeze(axis=1)
label = np.load(r'results\stock01_ns_Transformer_stock_ftM_sl30_ll0_pl1_dm512_nh8_el5_dl3_df2048_fc3_ebtimeF_dtTrue_Exp_stock04_0\true.npy').squeeze(axis=1)
pred_high = pred[:, 2]
label_high = label[:, 2]
x_range = np.arange(383)
plt.plot(x_range, pred_high, label='pred')
plt.plot(x_range, label_high, label='label')
plt.legend()
plt.show()

