import matplotlib.pyplot as plt
import os


def plot_training(history):

    os.makedirs("outputs", exist_ok=True)

    # Accuracy graph
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.savefig("outputs/training.png")
    plt.close()

    # Loss graph
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig("outputs/loss.png")
    plt.close()