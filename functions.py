import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input


def brighten_dark_regions(image, threshold=50, factor=1.5):
    """
    Brighten dark areas selectively.
    """
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < threshold
    brightened = image.copy()
    brightened[mask] = np.clip(brightened[mask] * factor, 0, 255).astype(np.uint8)
    return brightened


def adjust_brightness_contrast(image, contrast=1, brightness=50):
    """
    Adjusts brightness and contrast.
    Alpha > 1 increases contrast, beta > 0 increases brightness.
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


# Gradient ascent function for DeepDream
def deepdream(image, dream_model, iterations=2, step_size=.2):
    image = tf.convert_to_tensor(image)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = tf.reduce_mean(dream_model(image))
        grads = tape.gradient(loss, image)
        grads /= tf.math.reduce_std(grads) + 1e-8
        image = image + grads * step_size
    return image
# Prepare image for model
def preprocess_image(image, size):
    # Preprocess: Adjust brightness/contrast
    # image = adjust_brightness_contrast(image, contrast=0.5, brightness=50)
    image = tf.image.resize(image, (size, size))  # Resize for InceptionV3
    image = preprocess_input(image)
    return image

# Apply DeepDream effect
def apply_deepdream(frame, dream_model, img_size):
    # Preprocess frame
    input_image = preprocess_image(frame, img_size)
    input_image = tf.expand_dims(input_image, axis=0)
    # Apply deepdream
    dreamed_image = deepdream(input_image, dream_model)
    # Post-process and return
    dreamed_image = dreamed_image[0]
    dreamed_image = tf.image.resize(dreamed_image, frame.shape[:2])
    dreamed_image = tf.cast(dreamed_image, tf.uint8)
    return np.array(dreamed_image)


# Style transfer functions
def apply_style_transfer(content_image, style_image, style_transfer_model, image_size):
    # Preprocess images
    content_image = tf.image.resize(content_image, (image_size, image_size)) / 255.0
    style_image = tf.image.resize(style_image, (image_size, image_size)) / 255.0
    content_image = tf.expand_dims(content_image, axis=0)
    style_image = tf.expand_dims(style_image, axis=0)

    # Apply style transfer
    stylized_image = style_transfer_model(tf.constant(content_image), tf.constant(style_image))[0]
    return np.array(stylized_image[0] * 255, dtype=np.uint8)


# Psych functions
def color_invert(image):
    return cv2.bitwise_not(image)


def animated_ripple_effect(image, frame_count, amplitude=10, wavelength=30):
    h, w = image.shape[:2]
    x_indices = np.tile(np.arange(w), (h, 1))
    y_indices = np.tile(np.arange(h), (w, 1)).T

    amplitude = amplitude * np.sin(2 * np.pi * frame_count / 60)  # Amplitude oscillates
    freq = wavelength + 5 * np.sin(2 * np.pi * frame_count / 120)      # Frequency oscillates

    x_indices = (x_indices + amplitude * np.sin(2 * np.pi * y_indices / freq)).astype(np.float32)
    y_indices = (y_indices + amplitude * np.sin(2 * np.pi * x_indices / freq)).astype(np.float32)

    return cv2.remap(image, x_indices, y_indices, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def pulsating_brightness(image, frame_count, intensity=50):
    factor = 1 + 0.5 * np.sin(2 * np.pi * frame_count / 60)  # Oscillates between 0.5 and 1.5
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def animated_gradient_map(image, frame_count):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        lut[i, 0] = (
            (255 - i + frame_count * 5) % 256,  # Dynamic red channel
            (i * 2 + frame_count * 3) % 256,   # Dynamic green channel
            (i + frame_count * 7) % 256        # Dynamic blue channel
        )
    return cv2.LUT(image, lut)


def hue_shift(image, shift_value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_value) % 180  # Shift hue channel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def psychedelic_effect(image, frame_count, amplitude=3, wavelength=15, frame_count_div=4):
    # Apply hue shift
    hue_shifted = hue_shift(image, (frame_count) % 180)

    # Add ripple effect
    rippled = animated_ripple_effect(hue_shifted, frame_count=frame_count, amplitude=amplitude, wavelength=wavelength)

    # Apply gradient map
    psychedelic = animated_gradient_map(rippled, frame_count/frame_count_div)


    return psychedelic


# Cartoon functions
def add_math_effect(image, face_coords, text="E=mc^2"):
    h, w, _ = image.shape
    for face in face_coords:
        x_min = int(face[0] * w)
        y_min = int(face[1] * h)
        x_max = int(face[2] * w)
        y_max = int(face[3] * h)

        # Add trippy math equations and shapes
        cv2.putText(image, text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for i in range(10):
            center = (np.random.randint(x_min, x_max), np.random.randint(y_min, y_max))
            radius = np.random.randint(5, 15)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(image, center, radius, color, -1)

    return image


def quantize_colors(image, k=8):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and recreate the quantized image
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(image.shape)

    return quantized

def cartoonify(image):
    # Apply bilateral filter to smooth colors
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    # Quantize colors (reduce color palette)
    quantized = quantize_colors(image)

    # Combine quantized image with edges
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    return cartoon