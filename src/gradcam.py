import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


# =========================
# GRAD-CAM
# =========================
def get_gradcam(model, img_array, last_conv_layer_name):

    base_model = model.layers[0]

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


# =========================
# OVERLAY FUNCTION
# =========================
def overlay_heatmap(heatmap, img_path, alpha=0.4):

    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img

    # Save result
    cv2.imwrite("outputs/gradcam_result.jpg", superimposed_img)

    # Show result
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()