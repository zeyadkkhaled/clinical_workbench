# ui_manager.py
import tkinter.messagebox as messagebox
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image

import image_io
import spatial_engine
import morphology_engine
from config import UI_PADDING, CORNER_RADIUS

class UIManager:
    def __init__(self, master):
        self.master = master
        
        # State Management
        self.original_image_array = None
        self.current_image_array = None
        self.image_history = []
        self.metadata_dict = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure grid
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1) # Main canvas takes remaining space
        
        # 1. Left Sidebar
        self.sidebar_frame = ctk.CTkScrollableFrame(self.master, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        # 2. Main Canvas Frame
        self.canvas_frame = ctk.CTkFrame(self.master, corner_radius=0)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=UI_PADDING, pady=UI_PADDING)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.canvas_frame, text="No Image Loaded", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # 3. Right Panel (Metadata)
        self.right_panel = ctk.CTkFrame(self.master, width=250, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        
        self.metadata_title = ctk.CTkLabel(self.right_panel, text="DICOM Metadata", font=("Arial", 16, "bold"))
        self.metadata_title.pack(pady=(20, 10), padx=10)
        
        self.metadata_textbox = ctk.CTkTextbox(self.right_panel, width=230, wrap="word")
        self.metadata_textbox.pack(padx=10, pady=10, fill="both", expand=True)
        self.metadata_textbox.configure(state="disabled")
        
        self.build_sidebar_controls()
        
    def build_sidebar_controls(self):
        # -- I/O Controls --
        io_label = ctk.CTkLabel(self.sidebar_frame, text="I/O Controls", font=("Arial", 14, "bold"))
        io_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Load Image", command=self.on_load_image)
        self.btn_load.pack(pady=5, padx=10, fill="x")
        
        self.btn_save = ctk.CTkButton(self.sidebar_frame, text="Save Image", command=self.on_save_image)
        self.btn_save.pack(pady=5, padx=10, fill="x")
        
        # -- Zoom Controls --
        zoom_label = ctk.CTkLabel(self.sidebar_frame, text="Zoom Controls", font=("Arial", 14, "bold"))
        zoom_label.pack(pady=(20, 5), padx=10, anchor="w")
        
        self.scale_input = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Scale Factor (e.g. 2.0)")
        self.scale_input.pack(pady=5, padx=10, fill="x")
        
        self.interp_type_var = ctk.StringVar(value="Nearest Neighbor")
        self.interp_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, values=["Nearest Neighbor", "Linear"], variable=self.interp_type_var)
        self.interp_dropdown.pack(pady=5, padx=10, fill="x")
        
        self.btn_zoom = ctk.CTkButton(self.sidebar_frame, text="Apply Zoom", command=self.on_zoom)
        self.btn_zoom.pack(pady=5, padx=10, fill="x")
        
        # -- Spatial Filters --
        spatial_label = ctk.CTkLabel(self.sidebar_frame, text="Spatial Filters", font=("Arial", 14, "bold"))
        spatial_label.pack(pady=(20, 5), padx=10, anchor="w")
        
        self.kernel_size_input = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Kernel Size (odd, e.g. 3)")
        self.kernel_size_input.pack(pady=5, padx=10, fill="x")
        self.kernel_size_input.insert(0, "3")
        
        self.variance_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.1, to=5.0, number_of_steps=49)
        self.variance_slider.set(1.0)
        self.variance_slider.pack(pady=5, padx=10, fill="x")
        self.variance_label = ctk.CTkLabel(self.sidebar_frame, text="Gaussian Variance: 1.0")
        self.variance_label.pack(pady=0, padx=10)
        self.variance_slider.configure(command=lambda val: self.variance_label.configure(text=f"Gaussian Variance: {val:.1f}"))
        
        self.btn_avg = ctk.CTkButton(self.sidebar_frame, text="Average Filter", command=lambda: self.on_spatial_filter('average'))
        self.btn_avg.pack(pady=5, padx=10, fill="x")
        
        self.btn_gaussian = ctk.CTkButton(self.sidebar_frame, text="Gaussian Filter", command=lambda: self.on_spatial_filter('gaussian'))
        self.btn_gaussian.pack(pady=5, padx=10, fill="x")
        
        self.btn_median = ctk.CTkButton(self.sidebar_frame, text="Median Filter", command=self.on_median_filter)
        self.btn_median.pack(pady=5, padx=10, fill="x")
        
        self.btn_sobel = ctk.CTkButton(self.sidebar_frame, text="Sobel Edge", command=lambda: self.on_edge_detection('sobel'))
        self.btn_sobel.pack(pady=5, padx=10, fill="x")
        
        self.btn_prewitt = ctk.CTkButton(self.sidebar_frame, text="Prewitt Edge", command=lambda: self.on_edge_detection('prewitt'))
        self.btn_prewitt.pack(pady=5, padx=10, fill="x")
        
        # -- Contrast --
        contrast_label = ctk.CTkLabel(self.sidebar_frame, text="Contrast", font=("Arial", 14, "bold"))
        contrast_label.pack(pady=(20, 5), padx=10, anchor="w")
        
        self.block_size_input = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Block Size")
        self.block_size_input.pack(pady=5, padx=10, fill="x")
        
        self.btn_hist_eq = ctk.CTkButton(self.sidebar_frame, text="Apply Local Equalization", command=self.on_local_hist_eq)
        self.btn_hist_eq.pack(pady=5, padx=10, fill="x")
        
        # -- Morphology --
        morph_label = ctk.CTkLabel(self.sidebar_frame, text="Morphology", font=("Arial", 14, "bold"))
        morph_label.pack(pady=(20, 5), padx=10, anchor="w")
        
        self.se_shape_var = ctk.StringVar(value="Square")
        self.se_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, values=["Square", "Cross"], variable=self.se_shape_var)
        self.se_dropdown.pack(pady=5, padx=10, fill="x")
        
        self.se_size_input = ctk.CTkEntry(self.sidebar_frame, placeholder_text="SE Size (odd, e.g. 3)")
        self.se_size_input.pack(pady=5, padx=10, fill="x")
        self.se_size_input.insert(0, "3")
        
        self.btn_erode = ctk.CTkButton(self.sidebar_frame, text="Erosion", command=lambda: self.on_morphology('erode'))
        self.btn_erode.pack(pady=5, padx=10, fill="x")
        
        self.btn_dilate = ctk.CTkButton(self.sidebar_frame, text="Dilation", command=lambda: self.on_morphology('dilate'))
        self.btn_dilate.pack(pady=5, padx=10, fill="x")
        
        # -- Pipeline Controls --
        pipeline_label = ctk.CTkLabel(self.sidebar_frame, text="Pipeline Controls", font=("Arial", 14, "bold"))
        pipeline_label.pack(pady=(20, 5), padx=10, anchor="w")
        
        self.btn_undo = ctk.CTkButton(self.sidebar_frame, text="Undo Last Step", command=self.on_undo, fg_color="goldenrod")
        self.btn_undo.pack(pady=5, padx=10, fill="x")
        
        self.btn_reset = ctk.CTkButton(self.sidebar_frame, text="Reset to Original", command=self.on_reset, fg_color="maroon")
        self.btn_reset.pack(pady=5, padx=10, fill="x")

    def show_error(self, message):
        messagebox.showerror("Error", message)
        
    def show_info(self, message):
        messagebox.showinfo("Success", message)

    def update_canvas(self, image_array):
        """ Render NumPy array on CustomTkinter Canvas as an Image """
        if image_array is None:
            return
            
        # Ensure correct type for PIL
        if image_array.dtype != np.uint8:
            # Simple normalization to 0-255 if it's float or something else
            img_min = image_array.min()
            img_max = image_array.max()
            if img_max > img_min:
                norm_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                norm_array = np.zeros_like(image_array, dtype=np.uint8)
        else:
            norm_array = image_array

        pil_image = Image.fromarray(norm_array)
        
        # Calculate aspect ratio to fit the canvas frame nicely
        self.canvas_frame.update() # Get actual dimensions
        cf_width = self.canvas_frame.winfo_width()
        cf_height = self.canvas_frame.winfo_height()
        
        # Fallback if frame isn't fully drawn yet
        if cf_width <= 1 or cf_height <= 1:
            cf_width = 600
            cf_height = 600
            
        # Optional: scale down image if too large for display purposes
        img_width, img_height = pil_image.size
        ratio = min(cf_width / img_width, cf_height / img_height) * 0.95
        new_width = max(int(img_width * ratio), 1)
        new_height = max(int(img_height * ratio), 1)
        
        ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_width, new_height))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img # keep reference

    def update_metadata_panel(self, metadata_dict):
        self.metadata_textbox.configure(state="normal")
        self.metadata_textbox.delete("1.0", "end")
        
        if not metadata_dict:
            self.metadata_textbox.insert("end", "No metadata available.")
        else:
            for key, val in metadata_dict.items():
                self.metadata_textbox.insert("end", f"{key}:\n{val}\n\n")
                
        self.metadata_textbox.configure(state="disabled")

    def push_state(self):
        """ Save current state to history before changing """
        if self.current_image_array is not None:
            self.image_history.append(self.current_image_array.copy())
            
    def apply_action(self, action_func, *args, **kwargs):
        """ Generic wrapper to apply action, handle errors, and manage state """
        if self.current_image_array is None:
            self.show_error("No image loaded!")
            return
            
        try:
            self.push_state()
            new_image = action_func(self.current_image_array, *args, **kwargs)
            if new_image is None:
                raise ValueError("Engine returned None instead of an image array.")
            self.current_image_array = new_image
            self.update_canvas(self.current_image_array)
        except Exception as e:
            # Revert state push on failure and show error gracefully
            if self.image_history:
                self.image_history.pop()
            messagebox.showerror("Engine Error", str(e))

    # -- Callbacks --

    def on_load_image(self):
        path = filedialog.askopenfilename(title="Select Medical Image")
        if not path: return
        
        try:
            image_array, metadata, err = image_io.load_image(path)
            if err:
                self.show_error(err)
            else:
                self.original_image_array = image_array.copy()
                self.current_image_array = image_array.copy()
                self.image_history.clear()
                self.metadata_dict = metadata
                
                self.update_canvas(self.current_image_array)
                self.update_metadata_panel(self.metadata_dict)
        except Exception as e:
            self.show_error(f"Error loading image:\n{str(e)}")

    def on_save_image(self):
        if self.current_image_array is None:
            self.show_error("No image to save!")
            return
            
        path = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png")
        if not path: return
        
        try:
            success, msg = image_io.save_image(self.current_image_array, path)
            if success:
                self.show_info(msg)
            else:
                self.show_error(msg)
        except Exception as e:
            self.show_error(f"Error saving image:\n{str(e)}")

    def on_zoom(self):
        scale_str = self.scale_input.get()
        try:
            scale = float(scale_str)
        except ValueError:
            self.show_error("Invalid scale factor. Must be a number.")
            return
            
        interp = self.interp_type_var.get()
        if interp == "Nearest Neighbor":
            self.apply_action(spatial_engine.zoom_nearest_neighbor, scale)
        else:
            self.apply_action(spatial_engine.zoom_linear, scale)

    def on_spatial_filter(self, filter_type):
        try:
            k_size = int(self.kernel_size_input.get())
        except ValueError:
            self.show_error("Invalid kernel size. Must be an odd integer.")
            return
            
        if k_size % 2 == 0:
            self.show_error("Kernel size must be odd.")
            return
            
        var = self.variance_slider.get()
        self.apply_action(spatial_engine.apply_smoothing_filter, filter_type, k_size, variance=var)

    def on_median_filter(self):
        try:
            k_size = int(self.kernel_size_input.get())
        except ValueError:
            self.show_error("Invalid kernel size. Must be an odd integer.")
            return
            
        if k_size % 2 == 0:
            self.show_error("Kernel size must be odd.")
            return
            
        self.apply_action(spatial_engine.apply_median_filter, k_size)

    def on_edge_detection(self, operator_type):
        self.apply_action(spatial_engine.apply_edge_detection, operator_type)

    def on_local_hist_eq(self):
        try:
            b_size = int(self.block_size_input.get())
        except ValueError:
            self.show_error("Invalid block size. Must be an integer.")
            return
            
        self.apply_action(spatial_engine.local_histogram_equalization, b_size)

    def on_morphology(self, operation):
        try:
            size = int(self.se_size_input.get())
        except ValueError:
            self.show_error("Invalid SE size. Must be an odd integer.")
            return
            
        shape = self.se_shape_var.get()
        
        try:
            se = morphology_engine.generate_structuring_element(shape, size)
            if operation == 'erode':
                self.apply_action(morphology_engine.erode, se)
            else:
                self.apply_action(morphology_engine.dilate, se)
        except Exception as e:
            self.show_error(f"Error preparing morphology:\n{str(e)}")

    def on_undo(self):
        if not self.image_history:
            self.show_info("No more steps to undo.")
            return
            
        self.current_image_array = self.image_history.pop()
        self.update_canvas(self.current_image_array)

    def on_reset(self):
        if self.original_image_array is None:
            return
            
        self.push_state()
        self.current_image_array = self.original_image_array.copy()
        self.update_canvas(self.current_image_array)
