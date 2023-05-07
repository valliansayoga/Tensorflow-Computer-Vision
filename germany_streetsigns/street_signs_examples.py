from model import streetsigns_model
from utilities import create_generators
from keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    TRAIN = r"F:\Machine Learning\Tensorflow Computer Vision\data\training_data\train"
    VAL = r"F:\Machine Learning\Tensorflow Computer Vision\data\training_data\eval"
    TEST = r"F:\Machine Learning\Tensorflow Computer Vision\data\testing_data"
    BATCH_SIZE = 32
    EPOCHS = 3
    MODELS = "./Models"

    train_generator, val_generator, test_generator = create_generators(
        BATCH_SIZE, TRAIN, VAL, TEST
    )

    n_classes = train_generator.num_classes

    # To save the model
    checkpoint_saver = ModelCheckpoint(
        MODELS,
        monitor="val_accuracy",  # monitoring this value
        mode="max",  # Then save the model with highest val score
        save_best_only=True,  # replacing models with the best one
        save_freq="epoch",
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=2,  # Kalo sampai 2 epoch ga improve, stop!
    )

    model = streetsigns_model(n_classes)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_generator,
        callbacks=[
            checkpoint_saver,
            early_stop,
        ],
    )
