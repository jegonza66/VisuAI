import tkinter as tk
from tkinter import ttk, filedialog
import json
import os
import threading
import subprocess
from PIL import Image, ImageTk
import tensorflow as tf

class WebcamFilterUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VisuAI Controls")
        self.root.geometry("950x480")
        
        # Check GPU availability
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        # Load initial config
        self.load_config()
        
        # Create main frame with more padding
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure column weights to distribute space evenly
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)
        
        # Create logo frame at the top
        self.logo_frame = ttk.Frame(self.main_frame)
        self.logo_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 20))
        
        # Load and display logo
        try:
            logo_image = Image.open("visuai.png")
            # Resize logo to appropriate size (e.g., 150px wide)
            logo_image = logo_image.resize((150, int(150 * logo_image.height / logo_image.width)))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(self.logo_frame, image=self.logo_photo)
            logo_label.pack(side=tk.LEFT)
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        # Create three columns with more padding between them
        self.left_frame = ttk.Frame(self.main_frame, padding="10")
        self.middle_frame = ttk.Frame(self.main_frame, padding="10")
        self.right_frame = ttk.Frame(self.main_frame, padding="10")
        
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.middle_frame.grid(row=1, column=1, sticky="nsew", padx=10)
        self.right_frame.grid(row=1, column=2, sticky="nsew", padx=10)
        
        # Initialize all UI variables first
        self.style_path_var = tk.StringVar(value=self.config["style_image_path"])
        self.style_dir_var = tk.StringVar(value=self.config["style_images_dir"])
        self.randomize_var = tk.BooleanVar(value=self.config["randomize"])
        self.beats_var = tk.StringVar(value=str(self.config["beats"]))
        self.bpm_var = tk.StringVar(value=str(self.config["bpm"]))
        self.width_var = tk.StringVar(value=str(self.config.get("output_width", 1500)))
        self.height_var = tk.StringVar(value=str(self.config.get("output_height", 780)))
        self.dream_layer_var = tk.StringVar(value=self.config.get("dream_layer", "1"))
        self.img_load_size_var = tk.StringVar(value=str(self.config.get("img_load_size", 64)))
        self.save_output_var = tk.BooleanVar(value=bool(self.config.get("save_output_path", "")))
        self.use_gpu_var = tk.BooleanVar(value=self.gpu_available)  # Default to True if GPU available
        
        # Create all UI elements
        # Left Column: Models and basic Model Setup
        ttk.Label(self.left_frame, text="Models:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        # Model checkboxes
        self.model_vars = {}
        model_options = ["yolo", "style_transfer", "cyclegan_horse2zebra_pretrained", 
                        "cyclegan_style_vangogh_pretrained", "psych", "dream"]
        for i, model in enumerate(model_options):
            self.model_vars[model] = tk.BooleanVar(value=model in self.config.get("model_name", ""))
            ttk.Checkbutton(self.left_frame, text=model, variable=self.model_vars[model],
                          command=self.update_config).grid(row=i+1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Model Setup Parameters
        model_setup_frame = ttk.LabelFrame(self.left_frame, text="Model Setup", padding="5")
        model_setup_frame.grid(row=len(model_options)+2, column=0, columnspan=2, sticky="nsew", pady=(20, 5))
        
        ttk.Label(model_setup_frame, text="Image Load Size:").grid(row=0, column=0, sticky=tk.W)
        img_load_size_entry = ttk.Entry(model_setup_frame, textvariable=self.img_load_size_var, width=10)
        img_load_size_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        img_load_size_entry.bind('<KeyRelease>', self.update_config)
        
        # Dream Layer
        ttk.Label(model_setup_frame, text="Dream Layer:").grid(row=1, column=0, sticky=tk.W)
        dream_layer_combo = ttk.Combobox(model_setup_frame, textvariable=self.dream_layer_var, 
                                       values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], width=10)
        dream_layer_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        dream_layer_combo.bind('<<ComboboxSelected>>', self.update_config)
        
        # GPU Checkbox
        use_gpu_check = ttk.Checkbutton(model_setup_frame, text="Use GPU (if available)", 
                                      variable=self.use_gpu_var, command=self.update_config)
        use_gpu_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Middle Column: Style Controls and Timing
        ttk.Label(self.middle_frame, text="Style Image:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        style_entry = ttk.Entry(self.middle_frame, textvariable=self.style_path_var, width=40)  # Made wider
        style_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(self.middle_frame, text="Browse", command=self.browse_style_image).grid(row=1, column=1, sticky=tk.E)
        
        ttk.Label(self.middle_frame, text="Style Images Directory:", font=('Arial', 12, 'bold')).grid(row=2, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        style_dir_entry = ttk.Entry(self.middle_frame, textvariable=self.style_dir_var, width=40)  # Made wider
        style_dir_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(self.middle_frame, text="Browse", command=self.browse_style_dir).grid(row=3, column=1, sticky=tk.E)
        
        ttk.Checkbutton(self.middle_frame, text="Randomize Style", variable=self.randomize_var, 
                       command=self.update_config).grid(row=4, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        ttk.Label(self.middle_frame, text="Timing Controls:", font=('Arial', 12, 'bold')).grid(row=5, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        # Create a frame for timing controls to keep elements together
        timing_frame = ttk.Frame(self.middle_frame)
        timing_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(timing_frame, text="Beats:").grid(row=0, column=0, sticky=tk.W)
        beats_entry = ttk.Entry(timing_frame, textvariable=self.beats_var, width=10)
        beats_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))  # Added right padding
        beats_entry.bind('<KeyRelease>', self.update_config)
        
        ttk.Label(timing_frame, text="BPM:").grid(row=0, column=2, sticky=tk.W)
        bpm_entry = ttk.Entry(timing_frame, textvariable=self.bpm_var, width=10)
        bpm_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        bpm_entry.bind('<KeyRelease>', self.update_config)
        
        # Right Column: Output Settings
        ttk.Label(self.right_frame, text="Output Settings:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        
        # Resolution Section with bounding box
        resolution_frame = ttk.LabelFrame(self.right_frame, text="Resolution", padding="5")
        resolution_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 5))
        
        ttk.Label(resolution_frame, text="Width:").grid(row=0, column=0, sticky=tk.W)
        width_entry = ttk.Entry(resolution_frame, textvariable=self.width_var, width=10)
        width_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        width_entry.bind('<KeyRelease>', self.update_config)
        
        ttk.Label(resolution_frame, text="Height:").grid(row=1, column=0, sticky=tk.W)
        height_entry = ttk.Entry(resolution_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        height_entry.bind('<KeyRelease>', self.update_config)
        
        # Save Output Section with bounding box
        save_frame = ttk.LabelFrame(self.right_frame, text="Save Output", padding="5")
        save_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(20, 5))
        
        save_output_check = ttk.Checkbutton(save_frame, text="Save video", variable=self.save_output_var)
        save_output_check.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        save_output_check.bind('<Button-1>', self.update_config)

        # Run Button - Bottom Center
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        self.run_button = ttk.Button(self.button_frame, text="Run VisuAI", command=self.run_webcam_filter)
        self.run_button.pack()
        
        self.stop_button = ttk.Button(self.button_frame, text="Stop VisuAI", command=self.stop_webcam_filter)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack_forget()  # Hide stop button initially
        
        # Store widgets that should be disabled after running
        self.model_setup_widgets = [
            img_load_size_entry,
            use_gpu_check,
            dream_layer_combo,
            save_output_check
        ]
        
    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)[0]
                # Convert absolute paths to relative if they exist
                if 'style_image_path' in self.config:
                    self.config['style_image_path'] = os.path.relpath(self.config['style_image_path'])
                if 'style_images_dir' in self.config:
                    self.config['style_images_dir'] = os.path.relpath(self.config['style_images_dir'])
        except:
            self.config = {
                "model_name": "yolo",
                "style_image_path": "styles/style.jpg",
                "style_images_dir": "styles/",
                "bpm": 90,
                "beats": 32,
                "randomize": True,
                "img_load_size": 256,
                "gpu_ids": [0] if self.gpu_available else [],  # Use first GPU if available, empty list for CPU
                "save_output_path": "",
                "output_width": 1500,
                "output_height": 780,
                "dream_layer": "1"
            }
    
    def save_config(self):
        # Make a copy of the config to avoid modifying the original
        config_to_save = self.config.copy()
        # Convert paths to absolute before saving
        if 'style_image_path' in config_to_save:
            config_to_save['style_image_path'] = os.path.abspath(config_to_save['style_image_path'])
        if 'style_images_dir' in config_to_save:
            config_to_save['style_images_dir'] = os.path.abspath(config_to_save['style_images_dir'])
        with open('config.json', 'w') as f:
            json.dump([config_to_save], f, indent=2)
    
    def update_config(self, *args):
        # Get selected models
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        model_name = "+".join(selected_models) if selected_models else "none"
        
        self.config.update({
            "model_name": model_name,
            "style_image_path": self.style_path_var.get(),
            "style_images_dir": self.style_dir_var.get(),
            "bpm": int(self.bpm_var.get()) if self.bpm_var.get().isdigit() else 60,
            "beats": int(self.beats_var.get()) if self.beats_var.get().isdigit() else 4,
            "randomize": self.randomize_var.get(),
            "output_width": int(self.width_var.get()) if self.width_var.get().isdigit() else 1500,
            "output_height": int(self.height_var.get()) if self.height_var.get().isdigit() else 780,
            "dream_layer": self.dream_layer_var.get(),
            "img_load_size": int(self.img_load_size_var.get()) if self.img_load_size_var.get().isdigit() else 64,
            "gpu_ids": [0] if self.use_gpu_var.get() else [],  # Use first GPU if checked, empty list for CPU
            "save_output_path": "output/" if self.save_output_var.get() else ""
        })
        self.save_config()
    
    def browse_style_image(self):
        filename = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.style_path_var.get()),
            title="Select Style Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if filename:
            self.style_path_var.set(filename)
            self.update_config()
    
    def browse_style_dir(self):
        dirname = filedialog.askdirectory(
            initialdir=os.path.dirname(self.style_dir_var.get()),
            title="Select Style Images Directory"
        )
        if dirname:
            self.style_dir_var.set(dirname)
            self.update_config()
    
    def run_webcam_filter(self):
        # Disable the run button and model setup widgets while the filter is running
        self.run_button.config(state='disabled', text="Running...")
        self.run_button.pack(side=tk.LEFT, padx=5)  # Move run button to the side
        self.stop_button.pack(side=tk.LEFT, padx=5)  # Show stop button
        for widget in self.model_setup_widgets:
            widget.config(state='disabled')
        
        # Start the webcam filter in a separate process
        self.process = subprocess.Popen(['python', 'main.py', '--run'])
    
    def stop_webcam_filter(self):
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process = None
        
        # Re-enable the run button and model setup widgets
        self.run_button.config(state='normal', text="Run VisuAI")
        self.run_button.pack()  # Center run button again
        self.stop_button.pack_forget()  # Hide stop button
        for widget in self.model_setup_widgets:
            widget.config(state='normal')
    
    def run(self):
        self.root.mainloop()

def run_ui():
    ui = WebcamFilterUI()
    ui.run() 