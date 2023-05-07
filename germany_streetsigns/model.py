import tensorflow as tf
from keras import Model
from keras.layers import (
    Conv2D,
    Flatten,
    Input,
    Dense,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D
)

def streetsigns_model(n_classes):
    my_input = Input(
        shape=(60, 60, 3),  # 60x60 rough estimation; bisa hitung average W x H all imgs biar lebih bagus
    )
    
    x = Conv2D(32, (3, 3), activation="relu")(my_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    
    # PICK FLATTEN / GLOBAL AVG!
    # x = Flatten()(x) # NOTE : FLATTEN GA KOMPATIBLE DENGAN GLOBALAVG!
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, "relu")(x)
    x = Dense(n_classes, "softmax")(x)
    
    return Model(inputs=my_input, outputs=x)

if __name__ == "__main__":
    model = streetsigns_model(10)
    model.summary()