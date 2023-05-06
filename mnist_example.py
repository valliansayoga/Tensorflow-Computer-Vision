import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import (
    Conv2D,
    Input,
    Dense,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    GlobalAveragePooling2D,
)

#### Ada 3 Cara ####
# tensorf.ow.keras.Sequential
# ----- Common research architechture to double filters every time -----
model = tf.keras.Sequential(
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
# keras.Model: inherit from this class


def display_some_examples(images, labels):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        idx = np.random.randint(0, images.shape[0] - 1)
        img = images[idx]
        label = labels[idx]

        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data; gradient moves faster in most cases
    x_train = x_train.astype("float32") / 255  # Source data 8bit integer
    x_test = x_test.astype("float32") / 255  # Source data 8bit integer
    
    # One hot encoding label for y! Do if you want use categorical_crossentropy!
    # y_train = tf.keras.utils.to_categorical(y_train, 10)  # 10 labels
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    

    # Nambah dimensi karena di model ada dimensi tambahan yaitu n_channel
    # 3 karena nambah dimensi index ke 3 dari shape data nya
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # Model compiling
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # for classification
            # if categorical_crossentropy labels must 1 hot encoded!
        metrics="accuracy",
    )

    # Training
    model.fit(
        x_train, y_train, batch_size=64, epochs=3, validation_split=0.2
    )  # hyper parameters

    # Evaluation
    model.evaluate(x_test, y_test, batch_size=64)