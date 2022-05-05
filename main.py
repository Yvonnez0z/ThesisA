'''

@author: Jiayi Zhang 


'''

from abc import ABC
import scipy.io as scio
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, activations, Model, optimizers, metrics
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys
import os
from evaluation import evaluation

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
best_val_mae_output = sys.float_info.max
EPOCHS=100

class MyModel():
    def __init__(self, input_shape=(500, 56, 2)):
        inputs = layers.Input(shape=input_shape)

        self.cnn_layers = self.create_cnn(32, (5, 5), (1, 1), (64, 16), (32, 8)) + \
                          self.create_cnn(64, (3, 3), (1, 1), (32, 4), (16, 4)) + \
                          self.create_cnn(128, (2, 2), (1, 1), (4, 2), (8, 2))
        flatten_layer = layers.Flatten()
        self.branch_1 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(1, name='heart')]
        self.branch_2 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(1, name='breath')]
        outputs_1 = outputs_2 = inputs
        for layer in self.branch_1:
            outputs_1 = layer(outputs_1)
        for layer in self.branch_2:
            outputs_2 = layer(outputs_2)
        self.net = Model(inputs, [outputs_1, outputs_2])

    @staticmethod
    def create_cnn(filters=64, kernel_size=(2, 2), kernel_strides=(1, 1), pool_size=(2, 2), pool_strides=(1, 1)):
        module = [
            layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=kernel_strides, padding='same',
                          activation=activations.relu),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
        ]
        return module

    @staticmethod
    def create_fc(units=(128, 64, 32)):
        module = [layers.Dense(x, activation=activations.relu) for x in units]
        return module

def pre_processing(x, y, z):
    x = tf.cast((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), tf.float32)
    y = tf.squeeze(y)
    z = tf.squeeze(z)
    return x, (y, z)


def main():     
    z=np.load("1data.npz")
    data = z['arr_0']
    data1 = z['arr_2']
    data2 = z['arr_1']
      
    db_size = data.shape[0]
    batch_size = 4
    db = tf.data.Dataset.from_tensor_slices((data, data1, data2)).map(
        pre_processing).shuffle(10000)
    db_train = db.take(int(db_size * 0.8)).batch(batch_size)
    db_val = db.skip(int(db_size * 0.8)).batch(batch_size)

    model = MyModel()
    model.net.summary()
    checkpoint_filepath = './checkpoint1/optimal'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    model.net.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss=losses.MeanAbsolutePercentageError(name='MAPE'),
                      metrics=[metrics.MeanAbsolutePercentageError(name='MAPE'), 'mae'],
                      loss_weights=[0.5, 0.5])
    
    H = model.net.fit(db_train, epochs=EPOCHS, validation_data=db_val, callbacks=[model_checkpoint_callback])
    
    
    N=np.arange(0,EPOCHS) 
    plt.style.use("ggplot")    
    plt.figure()
    #plt.title('val_output_mae')
    plt.xlabel("Epoch #")
    plt.ylabel("Mean Absolute Percentage Error")
    #plt.plot(N, H.history["heart_MAPE"], label="Train Error1")
    plt.plot(N, H.history["val_heart_MAPE"], label="Mean Estimation Error of Heart")
    #plt.plot(N, H.history["breath_MAPE"], label="Train Error2")
    plt.plot(N, H.history["val_breath_MAPE"], label="Mean Estimation Error of Breath")
    plt.legend()
    plt.savefig('multiopt1.png')  
    np.save("multioutput_H_1.npy",H.history)

def predict():
    csi_sample = np.load("test_data.npy")
    heart_true = np.load("test_data1.npy")
    breath_true = np.load("test_data2.npy")
    
    
    sample = tf.data.Dataset.from_tensor_slices(csi_sample).map(
        lambda x: tf.cast((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), tf.float32)).batch(4)
    model = MyModel()
    model.net.load_weights('./checkpoint1/optimal').expect_partial()
    heart_pred, breath_pred = model.net.predict(sample)

    heart_mape = np.mean(np.abs(heart_pred - heart_true) / heart_true)
    breath_mape = np.mean(np.abs(breath_pred - breath_true) / breath_true)
    print(heart_mape, breath_mape)
    
    heart_mae=np.mean(np.abs(heart_pred - heart_true) )
    breath_mae=np.mean(np.abs(breath_pred - breath_true) )
    print(heart_mae, breath_mae)
 

    

    
    
if __name__ == '__main__':
    #main()
    #predict()
    evaluation()