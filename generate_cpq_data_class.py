import numpy as np
from scipy.stats import truncnorm
from matplotlib import pylab as plt


def dis(x,name='none'):
    x = np.array(x)
    print(name, " min: ", np.round(np.min(x),3), ", max: ", np.round(np.max(x),3), ", mean: ", np.round(np.mean(x),3), ", std: ",
          np.round(np.std(x),3), ", shape: ", x.shape)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def step(x):
    return (np.sign(x)+1)/2

def generate_graph(nr_of_layers,input_shape,output_shape=1, non_linear=np.sin):

    bias = [np.random.rand()*2 - 1 for i in range(nr_of_layers)]
    bias.append(np.random.rand(output_shape)*2 - 1)
    lintran = [np.random.rand(input_shape + i, 1) for i in range(nr_of_layers)]
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

    e = np.zeros((len(in_data),classes))
    r = np.int32(np.array(in_data)*classes)
    e[np.arange(len(in_data)), r] = 1
    return e, r

def one_hot(x, classes=[4,4,2,4,12,4]):

    x = np.array(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x,0)
    ret = np.zeros((len(x), np.sum(np.array(classes))))
    j = 0
    for i in range(len(classes)):
        ret[np.arange(len(x)),j+x[:,i]] = 1
        j += classes[i]
    ret = np.squeeze(ret)
    return ret

def back_transform(data, classes=[4,4,2,4,12,4]):

    #Make sure data is a numpy array
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data,0)
    ret = np.zeros((len(data),len(classes)))
    s = 0
    i=0
    for c in classes:
        # print("i : ", i, ", argmax: ", np.argmax(data[:,s:s+c],1))
        ret[:,i] = np.argmax(data[:,s:s+c],1)
        s += c
        i += 1
    return ret

def to_scalar(data, classes=[4,4,2,4,12,4]):

    #make sure data is numpy array and got 2 dims
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data,0)
    factor = np.array([np.prod(np.array(classes)[i+1:]) for i in range(len(classes))])
    # print("factor: ", factor)
    return np.int32(np.sum(data*factor, 1))

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

    def __init__(self, discount_max=0.4, discount_min=0, discount_mean=0.15, discount_std=0.1, mean_p=0.2, max_p=0.4):

        self.D_gen_base = lambda nr: truncnorm((discount_min - discount_mean) / discount_std,
                                          (discount_max - discount_mean) / discount_std,
                                          discount_mean,
                                          discount_std).rvs(nr)
        self.C_gen = C_gen
        self.P = PGenerator(C_combinations = C_gen(complete_set=True, copys=1), max_p = max_p, mean_p = mean_p)
        # self.P = PGenerator(C=None, D=None, max_p = 0.99, mean_p = 0.4)

        self.P_gen = self.P.sample
        self.nr = np.empty(6144,object)
        self.D_gen = np.empty(6144,object)

        self.copyfunction = lambda: 3


        C_combinations = self.C_gen(complete_set=True, copys=1)
        for C in C_combinations:
            self.nr[to_scalar(back_transform(C))] = lambda: self.copyfunction()
            self.D_gen[to_scalar(back_transform(C))] = lambda nr: self.D_gen_base(nr)

    def nr_of_sample(self, C): return self.nr[to_scalar(back_transform(C))][0]

    def distribution_function(self, C): return self.D_gen[to_scalar(back_transform(C))][0]

    def set_C(self, Ci, p=None, n=None, d_dist=None):

        #Make sure Ci is a numpy array
        Ci = np.array(Ci)
        #check if Ci is in format one hot (Ci.shape[-1]==30), if so transform back to standard form
        if Ci.shape[-1]==30:
            Ci = back_transform(Ci)

        id = to_scalar(Ci)[0]

        if p is not None:



            # make sure p is in range 0-1 for d 0-1
            dl = np.linspace(0,1,1001)
            p_ = p(dl)
            top = np.maximum(1,np.max(p_))
            bottom = np.minimum(0,np.min(p_))
            def p_norm(d): return np.minimum(1,np.maximum(0,(p(d) - bottom)/(top-bottom)))
            self.P.f[id] = p_norm
            self.P.basefunction[id] = False

        if n is not None:
            self.nr[id] = n

        if d_dist is not None:
            self.D_gen[id] = d_dist




    def generate_data(self, copyfunction=lambda: int(np.maximum(0,np.random.normal(100,2)))):

        C_data = []
        D_data = []
        self.copyfunction = copyfunction

        C_combinations = self.C_gen(nr=50000, complete_set=True, copys=1)

        for C in C_combinations:
            n = self.nr_of_sample(C)()
            C_data += [C]*n
            D_data += list(self.D_gen[to_scalar(back_transform(C))][0](n))
        C_data = np.array(C_data)
        D_data = np.array(D_data)

        p_data, deal_data = self.P.sample(C_data,D_data)

        shuffle_ind = np.arange(len(C_data))
        np.random.shuffle(shuffle_ind)
        C_data = C_data[shuffle_ind]
        D_data = D_data[shuffle_ind]
        p_data = p_data[shuffle_ind]
        deal_data = deal_data[shuffle_ind]

        return C_data, D_data, p_data, deal_data


class PGenerator:

    # If C is filled with data, D is also suppose to be filled with data and M anb b gets normalized to fit max_p and mean_p
    def __init__(self, C_combinations, max_p=0.4, mean_p=0.2):


        self.basefunction = np.array([True]*6144)
        self.f = np.empty(6144,object)
        self.com = 5
        dim = 6144
        # self.k = np.random.normal(5,20,(30,self.com)) + 10
        self.m = np.random.normal(0,0.05,(30,self.com)) + 0.2/6 # + np.ones((30,self.com)) * np.linspace(0,0.3,self.com)/6

        ampmin = 0
        ampmax = 40
        ampmean = 10
        ampstd = 5

        self.k = truncnorm((ampmin - ampmean) / ampstd,
                                          (ampmax - ampmean) / ampstd,
                                          ampmean,
                                          ampstd).rvs((30,self.com))

        self.amp = np.random.normal(0,0.4,(30,self.com))


        # print("self amp min: ", np.min(self.amp), ", max: ", np.max(self.amp))f
        # print("C combinations shape: ", C_combinations.shape)
        # print("C[0]: ", C_combinations[0])
        # aC_oh = C_combinations
        #
        # self.ks = np.matmul(aC_oh, Mk)
        # self.ms = np.matmul(aC_oh, Mk) * np.matmul(aC_oh, Mm)
        # amps = np.ones(dim) * max_p  # np.random.rand(dim)*p_max
        # ampmin=0
        # ampmax = 1
        # ampmean = 0.5
        # ampstd = 0.1
        #
        # ampbase = np.random.normal(0,1,(30,self.com))
        #
        # self.amps_p = np.zeros((dim, self.com))
        # for i in range(self.com):
        #     self.amps_p[:, i] = np.random.rand(dim) ** 4 * amps
        #     amps = amps - self.amps_p[:, i]
        # print("amps p shape: ", self.amps_p.shape, ", ks shape: ", self.ks.shape, "ms shape: ", self.ms.shape)

        self.ks = np.zeros((6144,self.com))
        self.ms = np.zeros((6144,self.com))
        self.amps_p = np.zeros((6144,self.com))

        for C in C_combinations:
            C_ind = to_scalar(back_transform(C))
            self.ks[C_ind] = np.matmul(C,self.k)
            self.ms[C_ind] = np.matmul(C,self.m)
            self.ms[C_ind] = self.ms[C_ind] * self.ks[C_ind]
            self.amps_p[C_ind] = sigmoid(np.matmul(C,self.amp))*1/self.com

        # f, ax = plt.subplots(2,2,figsize=(20,20))
        # ax[0,0].hist(np.reshape(self.ks,-1))
        # ax[0,0].set_title("k")
        # ax[0, 1].hist(np.reshape(self.ms/self.ks,-1))
        # ax[0, 1].set_title("m/k")
        # ax[1, 1].hist(np.reshape(self.amps_p,-1))
        # ax[1, 1].set_title("amps")
        # ax[1, 0].hist(np.reshape(self.ms,-1))
        # ax[1, 0].set_title("m")


        # print("self.amps_p min: ", np.min(self.amps_p), ", max: ", np.max(self.amps_p))
        # dis(self.ms, "ms")

    def base_func(self,C_ind, d):
        return np.sum(
            np.array([self.amps_p[C_ind, c] * sigmoid(self.ks[C_ind,c] * d - self.ms[C_ind,c]) for c in range(self.com)]), 0)

    def get_p_data(self,C,plot=False):

        if len(C) == 30:
            C = back_transform(C)

        if len(C) == 6:
            C = to_scalar(C)
        pp_tot = np.zeros(101)
        for c in range(self.com):
            print("part ", c, ": ", np.round(self.amps_p[C,c][0],3), "* sigmoid(", np.round(self.ks[C,c][0],3), " * d - ", np.round(self.ms[C,c][0],3),
                  ", center at d = ", np.round(self.ms[C,c][0]/self.ks[C,c][0],3))

            if plot:
                d = np.linspace(0,0.4,101)
                pp = self.amps_p[C,c]*sigmoid(self.ks[C,c] *d - self.ms[C,c])
                plt.plot(d,pp, label ='part '+str(c))
                pp_tot += pp
        if plot:
            d = np.linspace(0, 0.4, 101)
            plt.plot(d, pp_tot, lw=4, ls='--', color='black', label='total')
            plt.legend()
            plt.show()
            plt.close()

        # self.p = lambda C, d: np.array([self.f[to_scalar(back_transform(C_))][0]
        #                                 (to_scalar(back_transform(C_)),d_) for C_,d_ in zip(C,d)])

    def p(self,C,d,no_print=False):

        ret = []
        itt=0
        for C_,d_ in zip(C,d):

            if not no_print:
                if (itt+1)%1000 == 0:
                    print("Generated data: ", itt+1, "/", len(C), end="\r", flush=True)
                if itt == len(C)-1:
                    print("Generated data: ", itt+1, "/", len(C))

            C_ind = to_scalar(back_transform(C_))
            if self.basefunction[C_ind]:
                ret += [self.base_func(C_ind,d_)]
            else:
                # print("self.f[C_ind][0](d_): ", self.f[C_ind][0](d_))
                ret += [np.array([self.f[C_ind][0](d_)])]
                # print("ret: ", ret)
            itt+=1

        return np.array(ret)



    # defining p(C,D)
    def sample(self, C, D):

        p = np.squeeze(self.p(C,D))
        deal = np.random.binomial(1, p)
        # except:

            # print("p: ", np.min(p), " - ", np.max(p))

        return p, deal



