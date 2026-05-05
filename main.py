# Author: Zeyad Khaled (System Architect & Core Engine) - Complete Ownership
"""
Entry point for the Clinical Image Analysis Workbench.

Architecture Notes:
- State Management: The `UIManager` (instantiated below) manages all image state via 
  an `image_history` stack. Each morphological or spatial operation appends a fresh 
  NumPy array to this stack, allowing for step-by-step undo functionality.
- UI Garbage Collection Prevention: To display NumPy arrays via Tkinter, they are converted 
  to `ImageTk.PhotoImage`. `UIManager.refresh_canvas()` intentionally anchors these image 
  objects to a class variable (`self._photo_image`) to prevent Python's garbage collector 
  from prematurely deleting them and causing the canvas to render blank.
"""
import customtkinter as ctk
from config import APP_TITLE, WINDOW_GEOMETRY, THEME_COLOR, APPEARANCE_MODE
from ui_manager import UIManager

def main():
    ctk.set_appearance_mode(APPEARANCE_MODE)
    ctk.set_default_color_theme(THEME_COLOR)

    root = ctk.CTk()
    root.title(APP_TITLE)
    root.geometry(WINDOW_GEOMETRY)
    
    app = UIManager(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
