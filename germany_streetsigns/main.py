from model import streetsigns_model
from utilities import create_generators
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == "__main__":
    TRAIN = "./data/training_data/train"
    VAL = "./data/training_data/eval"
    TEST = "./data/testing_data"
    BATCH_SIZE = 32
    EPOCHS = 3
    MODELS = "./Models"
    TRAINING = True
    TESTING = True
    train_generator, val_generator, test_generator = create_generators(
        BATCH_SIZE, TRAIN, VAL, TEST
    )

    n_classes = train_generator.num_classes

    if TRAINING:
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
            # Optimizer can be from a class that have its own hyperparams
            # AMSGRAD in Adam is used when loss function is not changing
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

    if TESTING:
        model = tf.keras.models.load_model("./Models")
        model.summary()

        print("Evaluating on EVAL set", "-" * 80, sep="\n")
        model.evaluate(val_generator)

        print("Evaluating on TEST set", "-" * 80, sep="\n")
        model.evaluate(test_generator)
