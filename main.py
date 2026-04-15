from src.data_loader import load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.visualize import plot_training
from src.gradcam import get_gradcam, overlay_heatmap

import tensorflow as tf
import numpy as np
import cv2
import os

# =========================
# PATHS
# =========================
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"
model_path = "models/pneumonia_model.keras"

# =========================
# LOAD DATA
# =========================
train_data, val_data, test_data = load_data(train_dir, val_dir, test_dir)

print("\n================ CLASS INDICES CHECK ================")
print("Train:", train_data.class_indices)
print("Val:", val_data.class_indices)
print("Test:", test_data.class_indices)

# =========================
# LOAD OR TRAIN
# =========================
TRAIN_MODE = False   # 🔥 CHANGE THIS

if not os.path.exists(model_path) or TRAIN_MODE:
    print("\n⚠️ Training model...")

    model = build_model()

    history = train_model(model, train_data, val_data, epochs=3)

    # ✅ SAVE GRAPHS
    plot_training(history)

else:
    print("\n✅ Loading saved model...")
    model = tf.keras.models.load_model(model_path)

# =========================
# EVALUATION
# =========================
evaluate_model(model, test_data)

# =========================
# PREDICTION
# =========================
print("\nRunning sample prediction...")
predict_image(model, "data/sample/test1.jpeg")

# =========================
# GRAD-CAM
# =========================
print("\nGenerating Grad-CAM...")

img_path = "data/sample/test1.jpeg"
img = cv2.imread(img_path)

if img is not None:
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    _ = model.predict(img_array)

    heatmap = get_gradcam(model, img_array, "Conv_1")
    overlay_heatmap(heatmap, img_path)
else:
    print("Grad-CAM skipped: image not found.")