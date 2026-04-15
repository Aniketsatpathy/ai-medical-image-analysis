import numpy as np
import cv2

def predict_image(model, img_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.6:
        print(f"Pneumonia Detected ({prediction:.2f})")
    else:
        print(f"Normal ({1 - prediction:.2f})")