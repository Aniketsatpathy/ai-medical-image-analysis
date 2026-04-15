from tensorflow.keras.callbacks import ModelCheckpoint
import os


def train_model(model, train_data, val_data, epochs=5):

    os.makedirs("models", exist_ok=True)

    checkpoint = ModelCheckpoint(
        "models/pneumonia_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    return history