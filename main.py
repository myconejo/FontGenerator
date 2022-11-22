from Util.FontLoader import font_loader, font_single_loader
from Util.fontdataprovider import FontDataProvider
import numpy as np
import matplotlib.pyplot as plt


dp = FontDataProvider(".")

all = dp.all_list
train = dp.train_list
source = dp.source_list
valid = dp.val_list

print(train.shape, train[:10,0],train[:10,1],np.array(train[0][2]).shape)
print(valid.shape)
print(source.shape, source[:10,0],source[:10,1],np.array(source[0][2]).shape)
plt.figure(figsize=(1,2))
plt.imshow(np.array(source[0][2]),cmap='gray')
plt.show()


#font_path = "./GeneratedFontImage"
#font_single_loader("a",font_path,"arial",0, verbose=True)
#font_loader("arial", verbose=True)
#font_loader("arial")