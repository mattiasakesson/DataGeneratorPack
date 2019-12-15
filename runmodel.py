import tensorflow as tf
import tensorflow_probability as tfp
from generate_cpq_data_class import Generator_pack, one_hot, to_scalar
from matplotlib import pylab as plt
import numpy as np



plt.plot(np.random.rand(100))
plt.show()
plt.close

G=Generator_pack(discount_max = 0.4, discount_min=0, discount_mean = 0.15, discount_std = 0.2, mean_p = 0.2,
                  max_p = 0.4)

C_train, D_train, p_train, deal_train = G.generate_data(copyfunction=lambda: int(np.maximum(0,np.random.normal(10,0.2))))



tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

negloglik = lambda y, p_y: -p_y.log_prob(y)
eps = 0.000001
input_shape = (30,)
C = tf.keras.layers.Input(shape=input_shape)
dis = tf.keras.layers.Input(shape=(1,))

k_ = tf.keras.layers.Dense(10,name='k_layer')(C)
m_ = tf.keras.layers.Dense(10)(C)
z = tf.keras.layers.Lambda(lambda x: x[0]*x[1] - x[2],name='zlayer')([k_,dis,m_])
z2 = tf.keras.layers.Dense(1)(z)

Out = tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(probs=(1.0-eps)/(1. + tf.exp(-t) + eps/2)))(z2)

model = tf.keras.models.Model(inputs=[dis,C], outputs=Out)
k_model = tf.keras.models.Model(inputs=C, outputs=k_)
m_model = tf.keras.models.Model(inputs=C, outputs=m_)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01, decay=1e-4), loss=negloglik, metrics=['mae'])

model.summary()

model.fit([D_train,C_train], deal_train, batch_size=100, epochs=3, verbose=True)


C_1 = one_hot([1,1,1,1,1,1])
print("C_1: ", C_1)

nr=101
dl = np.linspace(0,0.4,nr)
C_1_list = np.tile(np.expand_dims(np.array(C_1),0),(nr,1))
print(C_1_list.shape)
p_pred = model([np.expand_dims(dl,1),C_1_list]).mean()
p_pred = np.squeeze(p_pred)
print("p_pred shape: ", p_pred.shape)
plt.plot(dl,p_pred)


true_C_1_p = lambda d: np.float64(G.P.base_func(to_scalar([1,1,1,1,1,1]), d))

true_p = true_C_1_p(dl)
print("true_p shape: ", true_p.shape)
# plt.plot(dl,true_p)
#
#
plt.show()
plt.close()