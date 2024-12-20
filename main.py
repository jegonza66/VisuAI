import cv2
import os
from datetime import datetime
import functions
import tensorflow as tf


#  Optimize tensorflow GPU usage
tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# style_transfer / cyclegan_horse2zebra_pretrained / cyclegan_style_vangogh_pretrained / yolo / psych / dream / cartoon
model_name = 'yolo_cyclegan_horse2zebra_pretrained'
img_load_size = 512  # only for style and dream models
output_width = 1550
output_height = 850
save_output_path = ''
dream_model_layer = '1'  # only for dream model
face_text = '',  # only for cartoon model
face_effects = '',  # only for cartoon model

# Define models
models, params = functions.define_models_params(model_name, img_load_size, output_width, output_height, save_output_path, dream_model_layer)

# initialize the camera (only one camera use port = 0)
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Get cam parameters
fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if save_output_path:
    print(f'saving output to {save_output_path} folder')
    os.makedirs(save_output_path, exist_ok=True)
    start_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    out = cv2.VideoWriter(f'{save_output_path}/{start_time}.mp4', fourcc, fps, (output_width, output_height))

# Run
frame_count = 0
while True:

    # Read frame
    ret, frame = cam.read()
    if not ret:
        raise ValueError('No camera detected')

    # Apply models
    if 'style_transfer' in model_name:
        frame = functions.transform_frame_style_transfer(models, frame, img_load_size, params['style_image_path'],
                                                         params['prev_style_image'], output_width, output_height)

    if 'cyclegan' in model_name:
        frame = functions.transform_frame_cyclegan(models, frame, img_load_size, params['opt'], params['transform'],
                                                   output_width, output_height)

    if 'yolo' in model_name:
        frame = functions.transform_frame_yolo(models, frame)

    if 'psych' in model_name:
        frame = functions.transform_frame_pych(frame, frame_count, amplitude=20, wavelength=75, frame_count_div=5)

    if 'dream' in model_name:
        frame = functions.transform_frame_dream(models, frame, img_load_size, output_width, output_height)

    # if 'cartoon' in model_name:
    #     frame = functions.transform_frame_cartoon(frame, face_effects, params['face_detection'], face_text)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Write the frame to the output file
    if save_output_path:
        out.write(frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    # Update frame count
    frame_count += 1

# Release camera and save file
cam.release()
if save_output_path:
    out.release()
cv2.destroyAllWindows()