import numpy as np
from matplotlib import pylab as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

def custom_loss_function(y, p_y):
    #logloss
    return -p_y.log_prob(y)

eps = 0.000001
input_shape = (30,)
print("updates?")

def create_model(input_shape=(30,)):
    C = tf.keras.layers.Input(shape=input_shape, name='C_data')
    dis = tf.keras.layers.Input(shape=(1,), name='D_data')

    k_ = tf.keras.layers.Dense(100, activation='relu', name='k_layer')(C)
    k_ = tf.keras.layers.Dense(100, name='k_layer2')(k_)

    m_ = tf.keras.layers.Dense(100, activation='relu', name='m_layer')(C)
    m_ = tf.keras.layers.Dense(100, name='m_layer2')(m_)

    z = tf.math.multiply(k_,dis, name='multiply_k')

    z = tf.math.add(z,m_,name='add_m')
    z = tf.keras.layers.Activation('sigmoid',name='h')(z)
    #z = tf.keras.layers.Dense(100,activation='sigmoid')(z)

    z = tf.keras.layers.Dense(1)(z)

    p = tf.keras.layers.Lambda(lambda t: tf.sigmoid(t) * (1.0 - eps) + eps / 2)(z)
    Out = tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(probs=t))(p)

    model = tf.keras.models.Model(inputs=[dis,C], outputs=Out)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, decay=0), loss=custom_loss_function, metrics=['mae'])

    return model

class MLmodel:

    def __init__(self, model_shell=create_model(), save_as="mlmodel"):

        self.model = model_shell
        self.save_as = save_as
        


    def train(self, C_train, D_train, deal_train, C_val, D_val, deal_val, batch_size=100,  verbose=0, early_stopping_patience=10):

        print("training starts")
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              verbose=verbose,
                                              patience=early_stopping_patience)

        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.save_as + '.hdf5',
                                                   save_best_only=True,
                                                   monitor='val_loss',
                                                   mode='min')

        val_data = ([D_val, C_val], deal_val)
        self.model.fit([D_train, C_train], deal_train,
                       validation_data=val_data,
                       callbacks=[es, mcp_save],
                       batch_size=batch_size,
                       epochs=500,
                       verbose=verbose)

    def probability(self, C_data, D_data):

        best_model = tf.keras.models.load_model(self.save_as + '.hdf5',
                                                custom_objects={'custom_loss_function': custom_loss_function})
        p_pred = np.squeeze(best_model([np.expand_dims(D_data,1), C_data]).mean())

        return p_pred






