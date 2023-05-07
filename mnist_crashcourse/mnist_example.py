import numpy as np
import tensorflow as tf

from deeplearning_models import (
    MyCustomModels,
    functional_model,
    sequential_model,
)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Models Init
    # model = sequential_model()
    # model = functional_model()
    model = MyCustomModels()

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
