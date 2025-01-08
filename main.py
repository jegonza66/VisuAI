import cv2
import os
from datetime import datetime
import functions
import tensorflow as tf
import json
# style_transfer / cyclegan_horse2zebra_pretrained / cyclegan_style_vangogh_pretrained / yolo / psych / dream


#  Optimize tensorflow GPU usage
tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define parameters
img_load_size = 64  # only for style and dream models
output_width = 1360
output_height = 768
save_output_path = ''
dream_model_layer = '1'  # only for dream model
gpu_ids = 0

# Define models
models, params = functions.define_models_params(img_load_size=img_load_size, output_width=output_width,
                                                output_height=output_height, save_output_path=save_output_path,
                                                dream_model_layer=dream_model_layer, gpu_ids=gpu_ids)

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

    # Load config
    with open('config.json') as f:
        try:
            config_dict = json.load(f)[0]
        except:
            pass
    model_name = config_dict['model_name']
    style_image_path = config_dict['style_image_path']
    face_text = config_dict['face_text']
    face_effects = config_dict['face_effects']

    # Read frame
    ret, frame = cam.read()
    if not ret:
        raise ValueError('No camera detected')

    # Apply models
    if 'cyclegan' in model_name:
        try:
            frame = functions.transform_frame_cyclegan(models=models, model_name=model_name, frame=frame, img_load_size=img_load_size,
                                                       opt=params['opt'], transform=params['transform'])
        except:
            pass

    if 'yolo' in model_name:
        try:
            frame = functions.transform_frame_yolo(models=models, frame=frame)
        except:
            pass

    if 'style_transfer' in model_name:
        try:
            frame, params['prev_style_image'] = (
            functions.transform_frame_style_transfer(models=models, frame=frame, img_load_size=img_load_size,
                                                     style_image_path=style_image_path,
                                                     prev_style_image=params['prev_style_image']))
        except:
            pass

    if 'psych' in model_name:
        try:
            frame = functions.transform_frame_pych(frame=frame, frame_count=frame_count, amplitude=20, wavelength=150, frame_count_div=3)
        except:
            pass

    if 'dream' in model_name:
        try:
            frame = functions.transform_frame_dream(models=models, frame=frame, img_load_size=img_load_size)
        except:
            pass

    # Resize output frame
    frame = cv2.resize(frame, (output_width, output_height))

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