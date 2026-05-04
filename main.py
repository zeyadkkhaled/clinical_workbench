# main.py
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
