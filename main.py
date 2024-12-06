import cv2
import torch
from options.test_options import TestOptions
from models import create_model
from torchvision import transforms
from util.util import tensor2im
import sys
import os
from datetime import datetime
import functions
import mediapipe as mp
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3


# Define model parameters
sys.argv = [
    'test.py',  # Script name
    '--name', 'style',  # horse2zebra_pretrained / style_vangogh_pretrained / cartoon / style / psych / dream
    '--load_size', '512',
    '--output_height', '512',
    '--output_width', '512',
    '--gpu_ids', '',  # 0 for
    '--no_dropout',
    '--dream_model_layer', '1',
    '--face_text', '',  # only for cartoon model
    '--face_effects', '',  # only for cartoon model
    '--save_output_path', ''
]

opt = TestOptions().parse() # Get default options
if opt.name == 'cartoon':
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
elif opt.name == 'style':
    # Load the pre-trained style transfer model
    style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    style_image_path = 'input/style.jpg'
    prev_style_image = cv2.imread(style_image_path)  # Replace with your style image path
    prev_style_image = cv2.cvtColor(prev_style_image, cv2.COLOR_BGR2RGB)
elif opt.name == 'dream':
    # Load pre-trained InceptionV3 model
    model = InceptionV3(include_top=False, weights='imagenet')
    dream_layer = model.get_layer(f'mixed{opt.dream_model_layer}')  # Layer to "dream" from
    dream_model = tf.keras.Model(inputs=model.input, outputs=dream_layer.output)
elif opt.name != 'psych':
    # Initialize CycleGAN model
    model = create_model(opt)  # Create the CycleGAN model
    model.setup(opt)  # Load the pre-trained weights

# Define transforms for webcam frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.load_size, opt.load_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# initialize the camera (only one camera use port = 0)
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Get cam parameters
fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if opt.save_output_path:
    print(f'saving output to {opt.save_output_path} folder')
    os.makedirs(opt.save_output_path, exist_ok=True)
    start_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    out = cv2.VideoWriter(f'{opt.save_output_path}/{start_time}.mp4', fourcc, fps, (opt.output_width, opt.output_height))

# Runq
frame_count = 0
while True:

    # Read frame
    ret, frame = cam.read()
    if not ret:
        raise ValueError('No camera detected')

    # Resize frame
    frame = cv2.resize(frame, (opt.load_size, opt.load_size))

    if opt.name == 'psych':
        output_frame = functions.psychedelic_effect(frame, frame_count, amplitude=20, wavelength=75, frame_count_div=5)

    elif opt.name == 'dream':
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply DeepDream effect (downscale for efficiency)
        dreamed_frame = functions.apply_deepdream(rgb_frame, dream_model, img_size=opt.load_size)
        # Convert back to BGR for OpenCV display
        output_frame = cv2.cvtColor(dreamed_frame, cv2.COLOR_RGB2BGR)

    elif opt.name == 'cartoon':
        output_frame = functions.cartoonify(frame)

        # Detect faces
        if opt.face_effects:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_coords = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_coords.append((bbox.xmin, bbox.ymin, bbox.xmin + bbox.width, bbox.ymin + bbox.height))

        # Add math effects
        if opt.face_text:
            output_frame = functions.add_math_effect(output_frame, face_coords, text=opt.face_text)

    elif opt.name == 'style':

        # Load new style image
        style_image = cv2.imread(style_image_path)  # Replace with your style image path
        # If loading failed, stick to previous style
        if not isinstance(style_image, np.ndarray):
            style_image = prev_style_image
        # If it worked, update previous style
        elif style_image.shape != prev_style_image.shape:
            prev_style_image = style_image
        elif (style_image == prev_style_image).any():
            prev_style_image = style_image

        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply style transfer
        styled_frame = functions.apply_style_transfer(rgb_frame, style_image, style_transfer_model, image_size=opt.load_size)

        # Convert back to BGR for OpenCV display
        output_frame = cv2.cvtColor(styled_frame, cv2.COLOR_RGB2BGR)

    else:
        # Convert to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform with model here
        input_tensor = transform(frame_rgb).unsqueeze(0).to(torch.device("cuda" if opt.gpu_ids else "cpu"))

        # Generate the transformed image
        model.set_input({'A': input_tensor})  # Set the input image
        model.test()  # Perform inference
        output_image = tensor2im(model.get_current_visuals()['fake'])

        # Convert the output image to a format suitable for OpenCV
        output_frame = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Resize output frame
    output_frame = cv2.resize(output_frame, (opt.output_width, opt.output_height))

    # Display the captured frame
    cv2.imshow('Camera', output_frame)

    # Write the frame to the output file
    if opt.save_output_path:
        out.write(output_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    # Update frame count
    frame_count += 1

# Release camera and save file
cam.release()
if opt.save_output_path:
    out.release()
cv2.destroyAllWindows()