from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_confusion_matrix(cm):
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


def evaluate_model(model, test_data):

    print("\nRunning Evaluation...\n")

    # Evaluate
    loss, acc = model.evaluate(test_data)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Predictions
    predictions = model.predict(test_data)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_data.classes

    # Sample debug
    print("\nSample Predictions vs Actual:")
    for i in range(10):
        print(f"Pred: {y_pred[i][0]} | Actual: {y_true[i]}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # ✅ SAVE IMAGE
    save_confusion_matrix(cm)