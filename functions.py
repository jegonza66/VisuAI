import os.path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import torch
from util.util import tensor2im
from ultralytics import YOLO
import sys
from options.test_options import TestOptions
import tensorflow_hub as hub
from torchvision import transforms
from models import create_model
import json
import random
import time


# Load config
with open('config.json') as f:
    config_dict = json.load(f)[0]
    # Convert paths to absolute for internal use
    style_image_path = os.path.abspath(config_dict['style_image_path'])
    style_transfer_model_path = os.path.abspath(config_dict['style_transfer_model_path'])

def read_config(config_dict, output_width, output_height, img_load_size, save_output_path, gpu_ids):
    """Reads the config file and updates parameters when changes are detected"""
    while True:
        try:
            with open('config.json') as f:
                new_config = json.load(f)[0]
                if new_config != config_dict:
                    config_dict.update(new_config)
                    # Update resolution if changed
                    if 'output_width' in new_config:
                        output_width = new_config['output_width']
                    if 'output_height' in new_config:
                        output_height = new_config['output_height']
        except:
            pass
        time.sleep(0.1)  # Check config every 100ms
    return config_dict, output_width, output_height

def define_models_params(img_load_size, output_width, output_height, save_output_path, gpu_ids):
    models = {}
    params = {}

    # ----- YOLO model ----- #
    # Load the YOLO11 model
    device = "cuda" if gpu_ids else "cpu"
    yolo_model = YOLO("checkpoints/yolo11n.pt")
    yolo_model.to(device)  # Move model to appropriate device
    models['yolo_model'] = yolo_model
    params['device'] = device

    # ----- Style transfer model ----- #
    # Load the pre-trained style transfer model
    style_transfer_model = hub.load(style_transfer_model_path)

    prev_style_image = cv2.imread(style_image_path)

    models['style_transfer_model'] = style_transfer_model
    params['prev_style_image'] = prev_style_image

    # ----- Cyclegans models ----- #
    # Define model parameters
    sys.argv = [
        'test.py',  # Script name
        '--no_dropout',
        '--gpu_ids', str(gpu_ids[0]) if gpu_ids else '',  # Pass first GPU ID or empty string for CPU
    ]

    # Define transforms for webcam frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_load_size, img_load_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load all models
    cyclegan_models = ['horse2zebra_pretrained', 'style_vangogh_pretrained']
    for model_name in cyclegan_models:
        opt = TestOptions().parse()  # Get default options
        opt.name = model_name
        opt.load_size = img_load_size
        opt.output_width = output_width
        opt.output_height = output_height
        opt.save_output_path = save_output_path

        # Initialize CycleGAN model
        cyclegan_model = create_model(opt)  # Create the CycleGAN model
        cyclegan_model.setup(opt)  # Load the pre-trained weights

        models[model_name] = cyclegan_model

    # Save transform and last opt (only used for gpu ids)
    params['opt'] = opt
    params['transform'] = transform

    return models, params

def randomize_style_image(style_images_dir, current_image=None):
    """
    Randomly selects a .jpg image from the specified directory, avoiding repetition of the current image.
    """
    # List all .jpg files in the directory
    jpg_files = [f for f in os.listdir(style_images_dir) if f.endswith('.jpg')]
    if not jpg_files:
        raise ValueError(f"No .jpg files found in directory: {style_images_dir}")

    # Remove the current image from the list if it exists
    if current_image in jpg_files:
        jpg_files.remove(current_image)

    # If only the current image exists, return it
    if not jpg_files:
        return os.path.join(style_images_dir, current_image)

    # Select a random .jpg file
    random_image = random.choice(jpg_files)
    return os.path.join(style_images_dir, random_image)

def transform_frame_style_transfer(models, frame, img_load_size, style_image_path, prev_style_image):

    # Unpack models
    style_transfer_model = models['style_transfer_model']

    # Resize frame
    frame = cv2.resize(frame, (img_load_size, img_load_size))

    # Load new style image
    if os.path.isfile(style_image_path):
        style_image = cv2.imread(style_image_path)
        # If loading failed, stick to previous style
        if not isinstance(style_image, np.ndarray):
            style_image = prev_style_image
        # If it worked, update previous style
        elif style_image.shape != prev_style_image.shape:
            prev_style_image = style_image
        elif (style_image != prev_style_image).any():
            prev_style_image = style_image
    else:
        # Load previous image
        style_image = prev_style_image

    # Convevrt color
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply style transfer
    frame = apply_style_transfer(frame, style_image, style_transfer_model, image_size=img_load_size)

    # Convert back to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame, prev_style_image

def transform_frame_cyclegan(models, model_name, frame, img_load_size, opt, transform):
    # Unpack model
    cyclegan_model_names = [model.replace('cyclegan_', '') for model in model_name.split('+') if 'cyclegan_' in model]
    cyclegan_models = [models[cyclegan_model_name] for cyclegan_model_name in cyclegan_model_names]

    # Resize frame
    frame = cv2.resize(frame, (img_load_size, img_load_size))

    # Convert to rgb
    input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for cyclegan_model in cyclegan_models:
        # Set up model input
        input_tensor = transform(input).unsqueeze(0).to(torch.device("cuda" if opt.gpu_ids else "cpu"))
        # Generate the transformed image
        cyclegan_model.set_input({'A': input_tensor})  # Set the input image
        cyclegan_model.test()  # Perform inference
        input = tensor2im(cyclegan_model.get_current_visuals()['fake'])

    # Convert the output image to a format suitable for OpenCV
    frame = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)

    return frame

def transform_frame_yolo(models, frame, device):

    # unpack model
    yolo_model = models['yolo_model']

    # Run YOLO11 prediction on the frame
    results = yolo_model.predict(frame, verbose=False, device=device)

    # Visualize the results on the frame
    frame = results[0].plot()

    return frame

def transform_frame_psych(frame, frame_count, amplitude=20, wavelength=75, frame_count_div=5):
    # Apply hue shift
    hue_shifted = hue_shift(frame, (frame_count) % 180)

    # Add ripple effect
    rippled = animated_ripple_effect(hue_shifted, frame_count=frame_count, amplitude=amplitude, wavelength=wavelength)

    # Apply gradient map
    psychedelic = animated_gradient_map(rippled, frame_count / frame_count_div)

    return psychedelic


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


# Prepare image for model
def preprocess_image(image, size):
    # Preprocess: Adjust brightness/contrast
    # image = adjust_brightness_contrast(image, contrast=0.5, brightness=50)
    image = tf.image.resize(image, (size, size))  # Resize for InceptionV3
    image = preprocess_input(image)
    return image


#----- Style transfer functions -----#
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