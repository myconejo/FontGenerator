import matplotlib.pyplot as plt
import torch
import os
import sys
cur_path = os.getcwd()
sys.path.insert(0,cur_path)
import numpy as np
from texttable import Texttable

loss_path_pre = os.path.join(cur_path,"results\\ckpt\\pre_train3")
loss_path_fine = os.path.join(cur_path,"results\\ckpt\\finetune")

loss_pre = torch.load(os.path.join(loss_path_pre, "losses.pt"))
loss_val = torch.load(os.path.join(loss_path_pre, "validation.pt"))
loss_fine = torch.load(os.path.join(loss_path_fine, "validation.pt"))
loss_pre=np.array(loss_pre).T
loss_val=np.array(loss_val)
loss_fine=np.array(loss_fine)

l1 = loss_pre[:,0]
l1_val = loss_val[:,0]
l1_fine = loss_fine[:,0]


n=96

l1 = [np.mean(l1[i*n:(i+1)*n]) for i in range(len(l1)//n)]
l1_val = [np.mean(l1_val[i:(i+1)]) for i in range(len(l1_val))]
l1_fine = [np.mean(l1_fine[i:(i+1)]) for i in range(len(l1_fine))]

tableon = Texttable()
tableon.set_cols_align(['l', 'c', 'c', 'c'])
tableon.set_cols_width(['15', '10', '10', '10'])
tableon.add_row(["","Train", "Validation", "Finetuning"])
tableon.add_row(["L1 loss(x100)",l1[-1]/5,l1_val[-1]/5, l1_fine[-1]/5])
print(tableon.draw())