import matplotlib.pyplot as plt
import torch
import os
import sys
cur_path = os.getcwd()
sys.path.insert(0,cur_path)
import numpy as np

loss_path = os.path.join(cur_path,"results\\ckpt\\pre_train")

loss = torch.load(os.path.join(loss_path, "losses.pt"))
loss=np.array(loss).T

l1 = loss[:,0]
const = loss[:,1]
cat = loss[:,2]
dloss = loss[:,3]
gloss = loss[:,4]

n=93
print(len(l1), len(dloss), len(gloss))

l1 = [np.mean(l1[i*n:(i+1)*n]) for i in range(len(l1)//n)]
dloss = [np.mean(dloss[i*n:(i+1)*n]) for i in range(len(dloss)//n)]
gloss = [np.mean(gloss[i*n:(i+1)*n]) for i in range(len(gloss)//n)]
index = range(len(l1))
print(len(l1), len(dloss), len(gloss))

plt.plot(index, l1, label = 'l1_loss')
plt.plot(index, dloss, label = 'GLoss')
plt.plot(index, gloss, label = 'DLoss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


