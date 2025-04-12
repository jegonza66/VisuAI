import tkinter as tk
from tkinter import ttk, filedialog
import json
import os
import threading
import subprocess
from PIL import Image, ImageTk  # Add PIL import for image handling

class WebcamFilterUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Webcam Filter Controls")
        self.root.geometry("1200x650")  # Slightly taller to accommodate logo
        
        # Load initial config
        self.load_config()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for main frame
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)  # Controls frame row
        
        # Create logo frame at the top
        self.logo_frame = ttk.Frame(self.main_frame)
        self.logo_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        # Load and display logo
        try:
            logo_image = Image.open("visuai.png")
            # Resize logo to appropriate size (e.g., 200px wide)
            logo_image = logo_image.resize((200, int(200 * logo_image.height / logo_image.width)))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(self.logo_frame, image=self.logo_photo)
            logo_label.pack(side=tk.LEFT)
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        # Create controls frame below logo
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure grid weights for controls frame
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)
        self.controls_frame.columnconfigure(2, weight=1)
        self.controls_frame.rowconfigure(0, weight=1)
        
        # Create three columns
        self.left_frame = ttk.Frame(self.controls_frame, padding="10")
        self.middle_frame = ttk.Frame(self.controls_frame, padding="10")
        self.right_frame = ttk.Frame(self.controls_frame, padding="10")
        
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.middle_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid(row=0, column=2, sticky="nsew")
        
        # Left Column: Models and Model Setup
        ttk.Label(self.left_frame, text="Models:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        self.model_vars = {}
        model_options = ["yolo", "style_transfer", "cyclegan_horse2zebra_pretrained", 
                        "cyclegan_style_vangogh_pretrained", "psych", "dream"]
        for i, model in enumerate(model_options):
            self.model_vars[model] = tk.BooleanVar(value=model in self.config.get("model_name", ""))
            ttk.Checkbutton(self.left_frame, text=model, variable=self.model_vars[model],
                          command=self.update_config).grid(row=i+1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Model Setup Parameters
        ttk.Label(self.left_frame, text="Model Setup:", font=('Arial', 12, 'bold')).grid(row=len(model_options)+2, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        ttk.Label(self.left_frame, text="Image Load Size:").grid(row=len(model_options)+3, column=0, sticky=tk.W)
        self.img_load_size_var = tk.StringVar(value=str(self.config.get("img_load_size", 64)))
        img_load_size_entry = ttk.Entry(self.left_frame, textvariable=self.img_load_size_var, width=10)
        img_load_size_entry.grid(row=len(model_options)+3, column=1, sticky=tk.W, padx=(5, 0))
        img_load_size_entry.bind('<KeyRelease>', self.update_config)
        
        ttk.Label(self.left_frame, text="GPU IDs:").grid(row=len(model_options)+4, column=0, sticky=tk.W)
        self.gpu_ids_var = tk.StringVar(value=str(self.config.get("gpu_ids", 0)))
        gpu_ids_entry = ttk.Entry(self.left_frame, textvariable=self.gpu_ids_var, width=10)
        gpu_ids_entry.grid(row=len(model_options)+4, column=1, sticky=tk.W, padx=(5, 0))
        gpu_ids_entry.bind('<KeyRelease>', self.update_config)
        
        # Middle Column: Style and Timing Controls
        # Style Image Path
        ttk.Label(self.middle_frame, text="Style Image:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        self.style_path_var = tk.StringVar(value=self.config["style_image_path"])
        style_entry = ttk.Entry(self.middle_frame, textvariable=self.style_path_var, width=30)
        style_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(self.middle_frame, text="Browse", command=self.browse_style_image).grid(row=1, column=1, sticky=tk.E)
        
        # Style Images Directory
        ttk.Label(self.middle_frame, text="Style Images Directory:", font=('Arial', 12, 'bold')).grid(row=2, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        self.style_dir_var = tk.StringVar(value=self.config["style_images_dir"])
        style_dir_entry = ttk.Entry(self.middle_frame, textvariable=self.style_dir_var, width=30)
        style_dir_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(self.middle_frame, text="Browse", command=self.browse_style_dir).grid(row=3, column=1, sticky=tk.E)
        
        # Randomize Style
        self.randomize_var = tk.BooleanVar(value=self.config["randomize"])
        ttk.Checkbutton(self.middle_frame, text="Randomize Style", variable=self.randomize_var, 
                       command=self.update_config).grid(row=4, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        # BPM and Beats Controls
        ttk.Label(self.middle_frame, text="Timing Controls:", font=('Arial', 12, 'bold')).grid(row=5, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        ttk.Label(self.middle_frame, text="BPM:").grid(row=6, column=0, sticky=tk.W)
        self.bpm_var = tk.StringVar(value=str(self.config["bpm"]))
        bpm_entry = ttk.Entry(self.middle_frame, textvariable=self.bpm_var, width=10)
        bpm_entry.grid(row=6, column=1, sticky=tk.W, padx=(5, 0))
        bpm_entry.bind('<KeyRelease>', self.update_config)
        
        ttk.Label(self.middle_frame, text="Beats:").grid(row=7, column=0, sticky=tk.W)
        self.beats_var = tk.StringVar(value=str(self.config["beats"]))
        beats_entry = ttk.Entry(self.middle_frame, textvariable=self.beats_var, width=10)
        beats_entry.grid(row=7, column=1, sticky=tk.W, padx=(5, 0))
        beats_entry.bind('<KeyRelease>', self.update_config)
        
        # Right Column: Face Controls, Resolution, and Output
        # Output Resolution
        ttk.Label(self.right_frame, text="Output Resolution:", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        
        ttk.Label(self.right_frame, text="Width:").grid(row=1, column=0, sticky=tk.W)
        self.width_var = tk.StringVar(value="1360")
        width_entry = ttk.Entry(self.right_frame, textvariable=self.width_var, width=10)
        width_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        width_entry.bind('<KeyRelease>', self.update_config)
        
        ttk.Label(self.right_frame, text="Height:").grid(row=2, column=0, sticky=tk.W)
        self.height_var = tk.StringVar(value="768")
        height_entry = ttk.Entry(self.right_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        height_entry.bind('<KeyRelease>', self.update_config)
        
        # Dream Model Layer
        ttk.Label(self.right_frame, text="Dream Layer:", font=('Arial', 12, 'bold')).grid(row=3, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        self.dream_layer_var = tk.StringVar(value="1")
        dream_layer_combo = ttk.Combobox(self.right_frame, textvariable=self.dream_layer_var, 
                                       values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], width=10)
        dream_layer_combo.grid(row=4, column=0, columnspan=2, sticky=tk.W)
        dream_layer_combo.bind('<<ComboboxSelected>>', self.update_config)
        
        # Save Output Path
        ttk.Label(self.right_frame, text="Save Output:", font=('Arial', 12, 'bold')).grid(row=5, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        self.save_output_var = tk.StringVar(value=self.config.get("save_output_path", ""))
        save_output_entry = ttk.Entry(self.right_frame, textvariable=self.save_output_var, width=30)
        save_output_entry.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))
        save_output_entry.bind('<KeyRelease>', self.update_config)
        
        # Run Button - Bottom Center
        self.run_button = ttk.Button(self.main_frame, text="Run Webcam Filter", command=self.run_webcam_filter)
        self.run_button.grid(row=2, column=0, pady=20)
        
        # Store widgets that should be disabled after running
        self.model_setup_widgets = [
            img_load_size_entry,
            gpu_ids_entry,
            dream_layer_combo,
            save_output_entry
        ]
        
    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)[0]
        except:
            self.config = {
                "model_name": "yolo",
                "style_image_path": "input/style.jpg",
                "style_images_dir": "input/styles/",
                "bpm": 60,
                "beats": 4,
                "randomize": True,
                "face_text": "Hola",
                "face_effects": True,
                "img_load_size": 64,
                "gpu_ids": 0,
                "save_output_path": ""
            }
    
    def save_config(self):
        with open('config.json', 'w') as f:
            json.dump([self.config], f, indent=2)
    
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
            "output_width": int(self.width_var.get()) if self.width_var.get().isdigit() else 1360,
            "output_height": int(self.height_var.get()) if self.height_var.get().isdigit() else 768,
            "dream_layer": self.dream_layer_var.get(),
            "img_load_size": int(self.img_load_size_var.get()) if self.img_load_size_var.get().isdigit() else 64,
            "gpu_ids": int(self.gpu_ids_var.get()) if self.gpu_ids_var.get().isdigit() else 0,
            "save_output_path": self.save_output_var.get()
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
        self.run_button.config(state='disabled')
        for widget in self.model_setup_widgets:
            widget.config(state='disabled')
        
        # Start the webcam filter in a separate process
        self.process = subprocess.Popen(['python', 'main.py', '--run'])
        
        # Add a button to stop the filter
        self.stop_button = ttk.Button(self.main_frame, text="Stop Webcam Filter", 
                                    command=self.stop_webcam_filter)
        self.stop_button.grid(row=2, column=0, pady=10)
    
    def stop_webcam_filter(self):
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process = None
        
        # Re-enable the run button and model setup widgets
        self.run_button.config(state='normal')
        for widget in self.model_setup_widgets:
            widget.config(state='normal')
        
        # Remove the stop button
        self.stop_button.destroy()
    
    def run(self):
        self.root.mainloop()

def run_ui():
    ui = WebcamFilterUI()
    ui.run() 