"""
main.py
Ownership: Zeyad
Purpose: The entry point that initializes the CustomTkinter app and runs the main loop.
"""
import customtkinter as ctk
from config import APP_TITLE, DEFAULT_WINDOW_SIZE
from ui_manager import UIManager

def main():
    # Set default theme and color
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = ctk.CTk()
    app.title(APP_TITLE)
    app.geometry(DEFAULT_WINDOW_SIZE)
    
    # Initialize the UI Manager
    ui = UIManager(app)
    
    # Start the application loop
    app.mainloop()

if __name__ == "__main__":
    main()
