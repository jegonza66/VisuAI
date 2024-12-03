import cv2
import torch
from options.test_options import TestOptions
from models import create_model
from torchvision import transforms
from util.util import tensor2im
import sys
import os
from datetime import datetime

start_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

# Define model parameters
sys.argv = [
    'test.py',  # Script name
    '--dataroot', 'datasets/vangogh2photo/testA',  # Path to your dataset
    '--name', 'horse2zebra_pretrained',  # Pretrained model
    '--model', 'test',  # Model type
    '--no_dropout',  # Disable dropout
    '--load_size', '512',
    '--crop_size', '512'
]

# Initialize CycleGAN model
opt = TestOptions().parse() # Get default options
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
os.makedirs('output/', exist_ok=True)
out = cv2.VideoWriter(f'output/{start_time}.mp4', fourcc, fps, (opt.crop_size, opt.crop_size))

# Run
while True:

    # Read frame
    ret, frame = cam.read()

    # Convert to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform with model here
    input_tensor = transform(frame_rgb).unsqueeze(0).to(torch.device("cuda" if opt.gpu_ids else "cpu"))

    # Generate the transformed image
    model.set_input({'A': input_tensor})  # Set the input image
    model.test()  # Perform inference
    output_image = tensor2im(model.get_current_visuals()['fake'])

    # Convert the output image to a format suitable for OpenCV
    output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display the captured frame
    cv2.imshow('Camera', output_bgr)

    # Write the frame to the output file
    out.write(output_bgr)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and save file
cam.release()
out.release()
cv2.destroyAllWindows()