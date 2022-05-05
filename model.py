
class MyModel:
    def __init__(self, input_shape=(1200, 56, 2)):
        inputs = layers.Input(shape=input_shape)
        self.cnn_layers = self.create_cnn(32, (5, 5), (1, 1), (40, 16), (20, 8)) + \
                          self.create_cnn(64, (3, 3), (1, 1), (20, 4), (10, 4)) + \
                          self.create_cnn(128, (2, 2), (1, 1), (12, 2), (6, 2))
        flatten_layer = layers.Flatten()
        self.branch_1 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(4)]
        outputs_1 = outputs_2 = inputs
        for layer in self.branch_1:
            outputs_1 = layer(outputs_1)

        self.net = Model(inputs, outputs_1)

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
    def create_fc(units=(64, 32)):
        module = [layers.Dense(x, activation=activations.relu) for x in units]
        return module