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
import pyvirtualcam
import numpy as np
import insightface


def visuai():
    # Read initial config
    with open('config.json', 'r') as f:
        config = json.load(f)[0]
    
    # Configure TensorFlow GPU usage based on gpu_ids
    if config.get('gpu_ids', []):  # If gpu_ids is not empty, use GPU
        tf.config.optimizer.set_jit(True)
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:  # If gpu_ids is empty, force CPU
        tf.config.set_visible_devices([], 'GPU')
    
    # Define models with initial config
    models, params = functions.define_models_params(
        img_load_size=config.get('img_load_size', 256),
        output_width=config.get('output_width', 1500),
        output_height=config.get('output_height', 780),
        save_output_path=config.get('save_output_path', 'output/'),
        gpu_ids=config.get('gpu_ids', [])  # Default to empty list (CPU)
    )

    # initialize the camera (only one camera use port = 0)
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    try:
        cam.getBackendName()
    except:
        raise ValueError('No camera detected')

    # Get cam parameters
    fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30

    # Define the codec and create VideoWriter object
    if config.get('save_output_bool'):
        output_path = config['save_output_path']
        print(f'saving output to {output_path} folder')
        os.makedirs(output_path, exist_ok=True)
        start_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
        # Use XVID codec instead of mp4v
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'{output_path}/{start_time}.avi', fourcc, fps, 
                            (config['output_width'], config['output_height']))

    # Setup virtual cam only if flag is True
    v_cam = None
    if config.get('use_virtual_cam'):
        v_cam = pyvirtualcam.Camera(width=config['output_width'], height=config['output_height'], fps=30)
        print("Virtual camera started")

    # Run
    frame_count = 0
    last_update_time = time.time()
    # Initialize tracking for the last beat timestamp received from the UI
    last_processed_beat_timestamp = None 
    config_dict = {}

    # Start config reading thread
    config_thread = threading.Thread(target=functions.read_config,
                                     args=(config_dict, config.get('output_width', 1500), config.get('output_height', 780), config.get('img_load_size', 256), config.get('save_output_path', 'output/'), config.get('gpu_ids', [0])), # Pass initial values correctly
                                     daemon=True)
    config_thread.start()
    
    # Wait a moment for the config thread to load the initial config
    time.sleep(0.5) 

    try:
        while True:
            # Read frame
            ret, frame = cam.read()
            if not ret:
                print('Camera frame read failed. Stopping.') # More informative message
                break # Exit cleanly if camera fails

            # --- Beat Sync Logic ---
            force_image_update = False # Flag to force update on beat mark
            current_beat_timestamp = config_dict.get("last_beat_timestamp", None)

            if current_beat_timestamp is not None and current_beat_timestamp != last_processed_beat_timestamp:
                print(f"Beat detected via UI at {current_beat_timestamp}, resetting timer.")
                last_update_time = current_beat_timestamp
                last_processed_beat_timestamp = current_beat_timestamp
                force_image_update = True # Signal to force update this frame
            # --- End Beat Sync Logic ---

            # Apply models in sequence
            model_name = config_dict.get('model_name', '')
            current_time = time.time() # Get current time for regular timer check

            # Calculate time since last update
            time_since_last_update = current_time - last_update_time
            # Calculate required interval (handle potential division by zero)
            bpm = config_dict.get('bpm', 60)
            beats = config_dict.get('beats', 4)
            update_interval = (beats * 60 / bpm) if bpm > 0 else float('inf') # Avoid division by zero

            # Check if it's time to update based on timer OR if forced by beat mark
            should_update_image = force_image_update or time_since_last_update >= update_interval

            if 'faceswap' in model_name and (config_dict.get('face_image_path', '') != ''
                                                   or isinstance(params.get('prev_face'), insightface.app.common.Face) # Use get for safety
                                                   or (config_dict.get('randomize_face', True) and config_dict.get('face_images_dir', '') != '')):
                try:
                    # Determine which face image path to use
                    face_image_path_to_use = params.get('prev_face_image_path', config_dict.get('face_image_path', '')) # Start with previous or config path
                    
                    if should_update_image:
                        print("Updating face image...") # Log update
                        if config_dict.get('randomize_face', True) and config_dict.get('face_images_dir', '') != '':
                            try:
                                # Pass the *current* base path to avoid selecting the same image if possible
                                current_face_basename = os.path.basename(params.get('prev_face_image_path', '')) if params.get('prev_face_image_path') else None
                                face_image_path_to_use = functions.randomize_face_image(
                                    face_images_dir=config_dict.get('face_images_dir', ''),
                                    current_image=current_face_basename)
                            except Exception as e_rand_face:
                                print(f"Could not randomize face: {e_rand_face}")
                                face_image_path_to_use = config_dict.get('face_image_path', '') # Fallback to selected path
                        else:
                            face_image_path_to_use = config_dict.get('face_image_path', '') # Use selected path if not randomizing
                        
                        last_update_time = current_time # Reset timer *after* deciding to update

                    # Only call transform if a valid path is determined
                    if face_image_path_to_use:
                         frame, params['prev_face'], params['prev_face_image_path'] = (
                            functions.transform_frame_faceswap(models=models, frame=frame,
                                                               face_detector=params['face_detector'],
                                                               face_image_path=face_image_path_to_use, # Use the determined path
                                                               prev_face_image_path= params.get('prev_face_image_path'), # Pass previous path
                                                               prev_face=params.get('prev_face'))) # Pass previous face data
                    else:
                        # If no path is available (e.g., first run, no random dir), maybe skip or log
                        pass 
                except Exception as e_fs: # Catch specific exceptions
                    print(f"Error during faceswap: {e_fs}")
                    pass

            if 'cyclegan' in model_name:
                try:
                    # NOTE: CycleGAN doesn't seem to have timed updates in the original code
                    frame = functions.transform_frame_cyclegan(models=models, model_name=model_name, frame=frame,
                                                             img_load_size=config_dict.get('img_load_size', 256),
                                                             opt=params['opt'], transform=params['transform'])
                except Exception as e_cg:
                    print(f"Error during CycleGAN: {e_cg}")
                    pass

            if 'yolo' in model_name:
                try:
                    # NOTE: YOLO doesn't seem to have timed updates
                    frame = functions.transform_frame_yolo(models=models, frame=frame, device=params['device'])
                except Exception as e_yolo:
                    print(f"Error during YOLO: {e_yolo}")
                    pass

            if 'style_transfer' in model_name and (config_dict.get('style_image_path', '') != ''
                                                   or isinstance(params.get('prev_style_image'), np.ndarray) # Use get for safety
                                                   or (config_dict.get('randomize_style', True) and config_dict.get('style_images_dir', '') != '')):
                try:
                    # Determine which style image path to use
                    style_image_path_to_use = params.get('prev_style_image_path', config_dict.get('style_image_path', '')) # Start with previous path or config path
                    
                    if should_update_image: # Use the combined update condition
                        print("Updating style image...") # Log update
                        if config_dict.get('randomize_style', True) and config_dict.get('style_images_dir', '') != '':
                            try:
                                # Pass the *current* base path
                                current_style_basename = os.path.basename(params.get('prev_style_image_path', '')) if params.get('prev_style_image_path') else None
                                style_image_path_to_use = functions.randomize_style_image(
                                    style_images_dir=config_dict.get('style_images_dir', ''),
                                    current_image=current_style_basename)
                            except Exception as e_rand_style:
                                print(f"Could not randomize style: {e_rand_style}")
                                style_image_path_to_use = config_dict.get('style_image_path', '') # Fallback
                        else:
                            style_image_path_to_use = config_dict.get('style_image_path', '') # Use selected path

                        last_update_time = current_time 

                    # Only call transform if a valid path is determined
                    if style_image_path_to_use:
                        frame, params['prev_style_image'] = (
                           functions.transform_frame_style_transfer(models=models, frame=frame,
                                                                    img_load_size=config_dict.get('img_load_size', 256),
                                                                    style_image_path=style_image_path_to_use, # Use determined path
                                                                    prev_style_image=params.get('prev_style_image'))) # Pass previous image data
                        # Store the path actually used for the next iteration's randomization check
                        params['prev_style_image_path'] = style_image_path_to_use 
                    else:
                         # If no path available, skip or log
                         pass
                except Exception as e_st: # Catch specific exceptions
                    print(f"Error during style transfer: {e_st}")
                    pass

            if 'psych' in model_name:
                try:
                    # NOTE: Psych doesn't seem to have timed updates
                    frame = functions.transform_frame_psych(frame=frame, frame_count=frame_count, amplitude=20, wavelength=150, frame_count_div=3)
                except Exception as e_psych:
                    print(f"Error during psych effect: {e_psych}")
                    pass

            # Resize output frame
            # Use .get() with defaults for robustness
            output_w = config_dict.get('output_width', 1500)
            output_h = config_dict.get('output_height', 780)
            if frame.shape[1] != output_w or frame.shape[0] != output_h: # Only resize if needed
                 try:
                     frame = cv2.resize(frame, (output_w, output_h))
                 except Exception as e_resize:
                     print(f"Error resizing frame: {e_resize}")
                     # Optionally keep the original frame or break

            # Display the captured frame
            cv2.imshow('VisuAI', frame)

            # Optionally route to virtual camera
            # Check config_dict for real-time toggle
            if config_dict.get('use_virtual_cam') and v_cam is not None:
                try:
                    # Brighten up
                    frame_vcam = functions.increase_brightness(img=frame, value=50)
                    frame_rgb = cv2.cvtColor(frame_vcam, cv2.COLOR_BGR2RGB)  # Convert RGB to RGB for OBS
                    frame_flipped = cv2.flip(frame_rgb, 1)
                    # Send the frame to the virtual camera
                    v_cam.send(frame_flipped)
                    v_cam.sleep_until_next_frame()
                except Exception as e_vcam:
                    print(f"Error sending frame to virtual cam: {e_vcam}")
                    # Consider stopping vcam or handling differently
                    v_cam = None # Stop trying if error occurs


            # Save frame if output path is set (Check config_dict for real-time toggle)
            if config_dict.get('save_output_bool') and 'out' in locals(): # Ensure 'out' is defined
                 try:
                     out.write(frame)
                 except Exception as e_save:
                     print(f"Error writing frame to video: {e_save}")
                     # Consider stopping saving

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting.")
                break

            # Update frame count
            frame_count += 1

    except KeyboardInterrupt:
         print("KeyboardInterrupt received, stopping.")
    except Exception as e_main_loop: # Catch other loop errors
         print(f"Error in main loop: {e_main_loop}")
    finally:
        # Clean up
        print("Cleaning up resources...")
        if 'cam' in locals() and cam.isOpened():
            cam.release()
        if 'out' in locals() and config_dict.get('save_output_bool'): # Check config again before releasing
            out.release()
        if 'v_cam' in locals() and v_cam is not None: # Close virtual cam if open
            v_cam.close()
        cv2.destroyAllWindows()
        print("Cleanup finished.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        visuai()
    else:
        run_ui()