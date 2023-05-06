import tensorflow as tf
from keras.layers import (
    Conv2D,
    Input,
    Dense,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
)

#### Ada 3 Cara ####
# tensorf.ow.keras.Sequential
# good for tutorial; easy to build
# ----- Common research architechture to double filters every time -----
sequential_model = tf.keras.Sequential(
    [
        Input(
            shape=(
                28,  # x pixels
                28,  # y pixels
                1,  # n_channels; grayscale hanya 1 channel
            )
        ),
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",  # rectified linear unit
        ),
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        ),
        MaxPooling2D(),
        BatchNormalization(),  # Helps minimises cost function
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
        ),
        MaxPooling2D(),
        BatchNormalization(),
        GlobalAveragePooling2D(),  # Hitung avg dari BatchNorm di atas
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),  # Output layer, 10 karena label angka 0-9
        # Softmax Outputs probability of sum 1. The highest prob is the prediction
    ]
)


# functional approach: function that returns a model
# ----- Every step is input for next step -----
# more flexible; can have multiple inputs/outputs
def functional_model():
    my_input = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(my_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=my_input, outputs=x)
    return model


# keras.Model: inherit from this class
# ----- More advanced; mirip pytorch ----
class MyCustomModels(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv2d1 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        self.conv2d2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
        self.maxpooling2d1 = MaxPooling2D()
        self.batchnormalization1 = BatchNormalization()

        self.conv2d3 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")
        self.maxpooling2d2 = MaxPooling2D()
        self.batchnormalization2 = BatchNormalization()

        self.globalaveragepooling2d1 = GlobalAveragePooling2D()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(10, activation="softmax")

    def call(self, my_input):
        x = self.conv2d1(my_input)
        x = self.conv2d2(x)
        x = self.maxpooling2d1(x)
        x = self.batchnormalization1(x)
        x = self.conv2d3(x)
        x = self.maxpooling2d2(x)
        x = self.batchnormalization2(x)
        x = self.globalaveragepooling2d1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
