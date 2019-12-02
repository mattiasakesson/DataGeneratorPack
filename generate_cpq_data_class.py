import numpy as np
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pylab as plt
from scipy.stats import truncnorm
import pickle





def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def step(x):
    return (np.sign(x)+1)/2






def generate_graph(nr_of_layers,input_shape,output_shape=1, non_linear=np.sin):

    bias = [np.random.rand()*2 - 1 for i in range(nr_of_layers)]
    bias.append(np.random.rand(output_shape)*2 - 1)
    lintran = [np.random.rand(input_shape + i, 1)  for i in range(nr_of_layers)]
    lintran.append(np.random.rand(input_shape + nr_of_layers, output_shape) )

    def graph(x_data):
        layers = [x for x in x_data.T]

        for i in range(nr_of_layers):
            new_layer = np.squeeze(non_linear(np.matmul(np.array(layers).T, lintran[i]) + bias[i]))
            layers.append(new_layer)

        layers = np.array(layers).T

        out_put = np.matmul(layers, lintran[nr_of_layers]) + bias[nr_of_layers]
        return out_put, layers
    return graph, bias, lintran

def to_cat(in_data,classes):
    print("in data range: ", np.min(in_data), " - ", np.max(in_data))
    e = np.zeros((len(in_data),classes))
    r = np.int32(np.array(in_data)*classes)
    print("r: ", r)

    e[np.arange(len(in_data)), r] = 1
    return e, r

def one_hot(x, classes=[4,4,2,4,12,4] ):

    x = np.array(x)
    # print("inside one_hot: x.shape: ", x.shape, ", type(x): ", type(x))
    if len(x.shape) == 1:
        x = np.expand_dims(x,0)
    ret = np.zeros((len(x), np.sum(np.array(classes))))
    j = 0
    for i in range(len(classes)):
        ret[np.arange(len(x)),j+x[:,i]] = 1
        j += classes[i]
    ret = np.squeeze(ret)
    # print("inside one_hot: ret type: ", type(ret), ", ret sample: ", ret[0:2])

    return ret

def C_gen(nr=50000, complete_set=False, copys=1):
    if complete_set:
        x_input = []
        for d1 in range(4):
            for d2 in range(4):
                for d3 in range(2):
                    for d4 in range(4):
                        for d5 in range(12):
                            for d6 in range(4):
                                x_input.append([d1, d2, d3, d4, d5, d6])

    else:
        dim = (nr, 1)
        x_input = np.concatenate((np.random.choice(4, dim),
                                  np.random.choice(4, dim),
                                  np.random.choice(2, dim),
                                  np.random.choice(4, dim),
                                  np.random.choice(12, dim),
                                  np.random.choice(4, dim)), 1)



    # Transform input data to one hot
    x_input_oh = one_hot(x_input)

    if complete_set and copys > 1:
        x_input_oh = np.array(list(x_input_oh)*int(copys))

    return x_input_oh

class Generator_pack:

       def __init__(self,discount_max = 0.4, discount_min=0, discount_mean = 0.15, discount_std = 0.1, mean_p = 0.2,
                  max_p = 0.4):
           self.D_gen = lambda nr: truncnorm((discount_min - discount_mean) / discount_std,
                                             (discount_max - discount_mean) / discount_std,
                                             discount_mean,
                                             discount_std).rvs(nr)
           self.C_gen = C_gen

           self.P = PGenerator(C=self.C_gen(50000), D=self.D_gen(50000), max_p = max_p, mean_p = mean_p)
           self.P_gen = self.P.sample








class PGenerator:
    # If C is filled with data, D is also suppose to be filled with data and M anb b gets normalized to fit max_p and mean_p
    def __init__(self, C=None, D=None, max_p=0.4, mean_p=0.2):

        self.M = np.random.rand(30, 2) * np.array([1, -1])
        self.b = np.random.rand(2) * np.array([1, -1])

        #Normalizing the distribution of p to fit max_p and mean_p
        if C is not None:

            y = np.matmul(C, self.M) + self.b
            z = y[:, 0] * D - y[:, 1]

            # finding m and k so mean(p)= mean_p and max(p) = max_p
            U = - np.log(1 / max_p - 1)
            fk = lambda m: (U - m) / np.max(z)
            m = 0
            delta_m = 0.5
            old_sign = np.sign(mean_p - np.mean(sigmoid(fk(m) * z + m)))
            i = 0
            while abs(mean_p - np.mean(sigmoid(fk(m) * z + m))) > 0.0001 and i < 10000:

                i += 1
                sign = np.sign(mean_p - np.mean(sigmoid(fk(m) * z + m)))
                if sign * old_sign == -1:
                    delta_m /= 2
                    old_sign = sign

                m += sign * delta_m

            k = fk(m)

            # Updating M and b to fit the desired transformation distribution
            self.M = self.M * k
            self.b = self.b * k
            self.b[1] = self.b[1] - m
            # y = np.matmul(C, self.M) + self.b
            # z = y[:, 0] * D - y[:, 1]
            # p = sigmoid(z)
            # print("p min: ", np.min(p), ", max: ", np.max(p), ", mean: ", np.mean(p))

    # defining p(C,D)
    def sample(self,C,D):
        y = np.matmul(C, self.M) + self.b
        z = y[:, 0] * D - y[:, 1]
        p = sigmoid(z)
        # print("p min: ", np.min(p), ", max: ", np.max(p))
        deal = np.random.binomial(1, p)

        return y, p, deal,



