import cv2
import os
from datetime import datetime
import time
import functions
import tensorflow as tf
import json
import threading
from ui import run_ui
import sys
# style_transfer / cyclegan_horse2zebra_pretrained / cyclegan_style_vangogh_pretrained / yolo / psych / dream

def run_webcam_filter():
    # Optimize tensorflow GPU usage
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Read initial config
    with open('config.json', 'r') as f:
        config = json.load(f)[0]
    
    # Define models with initial config
    models, params = functions.define_models_params(
        img_load_size=config.get('img_load_size', 64),
        output_width=config.get('output_width', 1360),
        output_height=config.get('output_height', 768),
        save_output_path=config.get('save_output_path', ''),
        dream_model_layer=config.get('dream_layer', '1'),
        gpu_ids=config.get('gpu_ids', 0)
    )

    # initialize the camera (only one camera use port = 0)
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    # Get cam parameters
    fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if config.get('save_output_path'):
        print(f'saving output to {config["save_output_path"]} folder')
        os.makedirs(config['save_output_path'], exist_ok=True)
        start_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
        out = cv2.VideoWriter(f'{config["save_output_path"]}/{start_time}.mp4', fourcc, fps, 
                            (config['output_width'], config['output_height']))

    # Run
    frame_count = 0
    # Get last update image time
    last_update_time = time.time()
    config_dict = {}

    # Start config reading thread
    config_thread = threading.Thread(target=functions.read_config, 
                                   args=(config_dict, config['output_width'], config['output_height'], 
                                        config['dream_layer'], models, params, config['img_load_size'], 
                                        config['save_output_path'], config['gpu_ids']),
                                   daemon=True)
    config_thread.start()

    while True:
        # Read frame
        ret, frame = cam.read()
        if not ret:
            raise ValueError('No camera detected')

        # Apply models in sequence
        selected_models = config_dict.get('model_name', '').split('+')
        
        for model_name in selected_models:
            if 'cyclegan' in model_name:
                try:
                    frame = functions.transform_frame_cyclegan(models=models, model_name=model_name, frame=frame, 
                                                             img_load_size=config_dict.get('img_load_size', 64),
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
                    current_time = time.time()
                    if current_time - last_update_time >= config_dict.get('beats', 4) / config_dict.get('bpm', 60):
                        if config_dict.get('randomize', True):
                            try:
                                style_image_path = functions.randomize_style_image(
                                    style_images_dir=config_dict.get('style_images_dir', ''), 
                                    current_image=os.path.basename(config_dict.get('style_image_path', '')))
                            except:
                                style_image_path = config_dict.get('style_image_path', '')
                        else:
                            style_image_path = config_dict.get('style_image_path', '')
                        last_update_time = current_time
                    frame, params['prev_style_image'] = (
                    functions.transform_frame_style_transfer(models=models, frame=frame, 
                                                           img_load_size=config_dict.get('img_load_size', 64),
                                                           style_image_path=style_image_path,
                                                           prev_style_image=params['prev_style_image']))
                except:
                    pass

            if 'psych' in model_name:
                try:
                    frame = functions.transform_frame_psych(frame=frame, frame_count=frame_count, amplitude=20, wavelength=150, frame_count_div=3)
                except:
                    pass

            if 'dream' in model_name:
                try:
                    frame = functions.transform_frame_dream(models=models, frame=frame, 
                                                          img_load_size=config_dict.get('img_load_size', 64))
                except:
                    pass

        # Resize output frame
        frame = cv2.resize(frame, (config_dict.get('output_width', 1360), config_dict.get('output_height', 768)))

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Write the frame to the output file
        if config_dict.get('save_output_path'):
            out.write(frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

        # Update frame count
        frame_count += 1

    # Release camera and save file
    cam.release()
    if config_dict.get('save_output_path'):
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_webcam_filter()
    else:
        run_ui()