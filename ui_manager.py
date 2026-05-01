"""
ui_manager.py
Ownership: Zeyad
Purpose: Build the main GUI (CustomTkinter) with a left sidebar, main canvas, and an "AI Segmentation" tab.
         Implement a state-management pipeline for Undo/Reset. Wire UI buttons.
"""
import customtkinter as ctk

# Import engines (currently skeletons)
import image_io
import spatial_engine
import frequency_engine
import morphology_engine
import ai_segmentation_engine

class UIManager:
    def __init__(self, master):
        self.master = master
        
        # State management pipeline
        self.image_history = []
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Builds the main GUI elements."""
        # Main layout configuration
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        
        # Left Sidebar for controls
        self.sidebar_frame = ctk.CTkFrame(self.master, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Workbench Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Example buttons wiring to engines
        self.load_btn = ctk.CTkButton(self.sidebar_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.undo_btn = ctk.CTkButton(self.sidebar_frame, text="Undo", command=self.undo_action)
        self.undo_btn.grid(row=2, column=0, padx=20, pady=10)
        
        # Main Canvas Area
        self.main_frame = ctk.CTkFrame(self.master)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # TODO (Zeyad): Bind canvas events for ROI and Notch filtering here
        # Example: self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Tabs for different processing areas including "AI Segmentation"
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.tabview.add("Spatial")
        self.tabview.add("Frequency")
        self.tabview.add("Morphology")
        self.tabview.add("AI Segmentation")
        
        # Setup specific tab UI elements and bind to engine functions...
        
    def load_image(self):
        """Handles loading an image and updating history."""
        # TODO (Zeyad): Open file dialog, call image_io.load_image, update current_image and history
        pass
        
    def undo_action(self):
        """Restores the previous image state."""
        # TODO (Zeyad): Implement popping from image_history and updating display
        pass
