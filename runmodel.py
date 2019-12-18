from generate_cpq_data_class import Generator_pack, one_hot, to_scalar
from matplotlib import pylab as plt
import numpy as np
from MLmodel import*
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    return -np.log(p*true_label + (1-p)*(1- true_label))


G=Generator_pack(discount_max = 0.4, discount_min=0, discount_mean = 0.15, discount_std = 0.2, mean_p = 0.2,
                  max_p = 0.4)

C_train, D_train, p_train, deal_train = G.generate_data(copyfunction=lambda: int(np.maximum(0,np.random.normal(10,0.2))))
C_val, D_val, p_val, deal_val = G.generate_data(copyfunction=lambda: int(np.maximum(0,np.random.normal(10,0.2))))


print("true logloss train set: ", np.mean(logloss(deal_train,p_train)))
print("true logloss test set: ", np.mean(logloss(deal_val,p_val)))

M = MLmodel()
M.train(C_train,D_train, deal_train, C_val, D_val, deal_val,verbose=1)

p_val = M.probability(C_val,D_val)

C_1 = [1,1,1,1,1,1]
ind = np.where([list(C) == C_1 for C in C_val])
plt.scatter(D_val[ind], p_val[ind])