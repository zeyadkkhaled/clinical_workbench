# ui_manager.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image

import image_io
import spatial_engine
import morphology_engine

from config import (
    UI_PADDING, CORNER_RADIUS, SIDEBAR_WIDTH,
    CLR_BG_MAIN, CLR_BG_SIDEBAR, CLR_BG_CARD, CLR_BG_CARD_HDR, CLR_BORDER,
    CLR_ACCENT, CLR_ACCENT_HOVER, CLR_ACCENT_DIM,
    CLR_CYAN, CLR_CYAN_HOVER,
    CLR_TEXT_PRI, CLR_TEXT_SEC, CLR_TEXT_HDG,
    CLR_SUCCESS, CLR_SUCCESS_HVR,
    CLR_WARNING, CLR_WARNING_HVR,
    CLR_DANGER, CLR_DANGER_HVR,
    CLR_ST_IDLE, CLR_ST_SUCCESS, CLR_ST_WARNING, CLR_ST_ERROR, CLR_ST_BUSY,
    FONT_APP_TITLE, FONT_SECTION_HDG, FONT_LABEL, FONT_LABEL_BOLD,
    FONT_BUTTON, FONT_MONO, FONT_STATUS,
)

_MAX_DISPLAY_PX = 1600   # longest display edge; original array is never downscaled


class UIManager:
    def __init__(self, master):
        self.master = master
        self.master.minsize(1100, 700)

        self.original_image_array = None
        self.current_image_array  = None
        self.image_history        = []
        self.metadata_dict        = {}
        self._current_tk_image    = None   # GC anchor

        self.setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def setup_ui(self):
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_columnconfigure(0, weight=0)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=0)

        self._build_sidebar()
        self._build_viewer()
        self._build_right_panel()
        self._build_status_bar()
        self.build_sidebar_controls()

    def _build_sidebar(self):
        self.sidebar_frame = ctk.CTkScrollableFrame(
            self.master,
            width=SIDEBAR_WIDTH,
            corner_radius=0,
            fg_color=CLR_BG_SIDEBAR,
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        title_strip = ctk.CTkFrame(self.sidebar_frame, fg_color=CLR_ACCENT_DIM, corner_radius=8)
        title_strip.pack(fill="x", padx=8, pady=(8, 12))
        ctk.CTkLabel(
            title_strip,
            text="⚕  Clinical Workbench",
            font=FONT_APP_TITLE,
            text_color=CLR_TEXT_HDG,
            anchor="w",
        ).pack(fill="x", padx=12, pady=8)

    def _build_viewer(self):
        self.viewer_frame = ctk.CTkFrame(
            self.master, corner_radius=0, fg_color=CLR_BG_MAIN,
        )
        self.viewer_frame.grid(row=0, column=1, sticky="nsew")

        self.image_label = ctk.CTkLabel(
            self.viewer_frame,
            text="No image loaded\n\nPress  Load Image  in the sidebar to begin",
            font=("Segoe UI", 14),
            text_color=CLR_TEXT_SEC,
            anchor="center",
        )
        self.image_label.pack(expand=True, fill="both")

    def _build_right_panel(self):
        self.right_panel = ctk.CTkFrame(
            self.master, width=250, corner_radius=0, fg_color=CLR_BG_SIDEBAR,
        )
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.right_panel.grid_propagate(False)

        hdr = ctk.CTkFrame(self.right_panel, fg_color=CLR_BG_CARD_HDR, corner_radius=0, height=42)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text="  Image Info",
            font=FONT_SECTION_HDG, text_color=CLR_TEXT_HDG, anchor="w",
        ).pack(fill="both", expand=True, padx=10)

        stats_card = ctk.CTkFrame(self.right_panel, fg_color=CLR_BG_CARD, corner_radius=8)
        stats_card.pack(fill="x", padx=8, pady=(10, 4))
        self.stats_label = ctk.CTkLabel(
            stats_card,
            text="No image loaded",
            font=FONT_MONO,
            text_color=CLR_TEXT_SEC,
            anchor="w",
            justify="left",
        )
        self.stats_label.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            self.right_panel, text="  Metadata",
            font=FONT_LABEL_BOLD, text_color=CLR_TEXT_SEC, anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.metadata_textbox = ctk.CTkTextbox(
            self.right_panel,
            font=FONT_MONO,
            fg_color=CLR_BG_CARD,
            text_color=CLR_TEXT_SEC,
            corner_radius=8,
            wrap="word",
            activate_scrollbars=True,
        )
        self.metadata_textbox.pack(padx=8, pady=(0, 4), fill="both", expand=True)
        self.metadata_textbox.configure(state="disabled")

        self.history_label = ctk.CTkLabel(
            self.right_panel,
            text="Undo steps: 0",
            font=FONT_LABEL,
            text_color=CLR_TEXT_SEC,
        )
        self.history_label.pack(pady=(0, 8))

    def _build_status_bar(self):
        self.status_bar = ctk.CTkFrame(
            self.master, corner_radius=0, fg_color=CLR_ST_IDLE, height=30,
        )
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status_bar.grid_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="  Ready",
            font=FONT_STATUS,
            text_color=CLR_TEXT_SEC,
            anchor="w",
        )
        self.status_label.pack(side="left", padx=4)

        self.dim_label = ctk.CTkLabel(
            self.status_bar,
            text="",
            font=FONT_MONO,
            text_color=CLR_TEXT_SEC,
            anchor="e",
        )
        self.dim_label.pack(side="right", padx=8)

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar widget helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_card(self, title):
        """Returns the content frame of a new sidebar section card."""
        outer = ctk.CTkFrame(self.sidebar_frame, fg_color=CLR_BG_CARD, corner_radius=10)
        outer.pack(fill="x", padx=8, pady=(0, 8))

        hdr = ctk.CTkFrame(outer, fg_color=CLR_BG_CARD_HDR, corner_radius=7, height=28)
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text=f"  {title}",
            font=FONT_SECTION_HDG, text_color=CLR_TEXT_HDG, anchor="w",
        ).pack(fill="both", expand=True, padx=6)

        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.pack(fill="x", padx=6, pady=(4, 6))
        return content

    def _muted_label(self, parent, text):
        lbl = ctk.CTkLabel(
            parent, text=text, font=FONT_LABEL, text_color=CLR_TEXT_SEC, anchor="w",
        )
        lbl.pack(fill="x", pady=(5, 0))
        return lbl

    def _styled_entry(self, parent, placeholder="", default=""):
        e = ctk.CTkEntry(
            parent,
            placeholder_text=placeholder,
            font=FONT_LABEL,
            fg_color=CLR_BG_MAIN,
            border_color=CLR_BORDER,
            corner_radius=6,
            height=30,
        )
        e.pack(fill="x", pady=2)
        if default:
            e.insert(0, default)
        return e

    def _styled_optionmenu(self, parent, values, variable):
        om = ctk.CTkOptionMenu(
            parent,
            values=values,
            variable=variable,
            font=FONT_LABEL,
            fg_color=CLR_BG_CARD_HDR,
            button_color=CLR_ACCENT_DIM,
            button_hover_color=CLR_ACCENT,
            dropdown_fg_color=CLR_BG_CARD,
            corner_radius=6,
        )
        om.pack(fill="x", pady=2)
        return om

    def _op_btn(self, parent, text, cmd, fg=None, hover=None):
        btn = ctk.CTkButton(
            parent,
            text=text,
            command=cmd,
            font=FONT_BUTTON,
            fg_color=fg or CLR_ACCENT,
            hover_color=hover or CLR_ACCENT_HOVER,
            corner_radius=6,
            height=32,
        )
        btn.pack(fill="x", pady=2)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    # Build all sidebar sections
    # ─────────────────────────────────────────────────────────────────────────

    def build_sidebar_controls(self):
        # I/O Controls
        c = self._make_card("I / O  Controls")
        self.btn_load = self._op_btn(c, "Load Image",  self.on_load_image,  fg=CLR_SUCCESS,  hover=CLR_SUCCESS_HVR)
        self.btn_save = self._op_btn(c, "Save Image",  self.on_save_image,  fg=CLR_CYAN,     hover=CLR_CYAN_HOVER)

        # Zoom Controls
        c = self._make_card("Zoom Controls")
        self._muted_label(c, "Scale Factor")
        self.scale_input = self._styled_entry(c, placeholder="e.g. 1.5")
        self._muted_label(c, "Interpolation Method")
        self.interp_type_var = ctk.StringVar(value="Nearest Neighbor")
        self.interp_dropdown = self._styled_optionmenu(c, ["Nearest Neighbor", "Linear"], self.interp_type_var)
        self.btn_zoom = self._op_btn(c, "Apply Zoom", self.on_zoom)

        # Spatial Filters
        c = self._make_card("Spatial Filters")
        self._muted_label(c, "Kernel Size  (odd integer)")
        self.kernel_size_input = self._styled_entry(c, default="3")
        self._muted_label(c, "Gaussian Variance")
        self.variance_slider = ctk.CTkSlider(
            c, from_=0.1, to=5.0, number_of_steps=49,
            progress_color=CLR_ACCENT,
            button_color=CLR_TEXT_HDG,
            button_hover_color=CLR_TEXT_PRI,
        )
        self.variance_slider.set(1.0)
        self.variance_slider.pack(fill="x", pady=(2, 0))
        self.variance_label = ctk.CTkLabel(
            c, text="Variance: 1.0", font=FONT_LABEL, text_color=CLR_TEXT_SEC,
        )
        self.variance_label.pack()
        self.variance_slider.configure(
            command=lambda v: self.variance_label.configure(text=f"Variance: {v:.1f}")
        )
        self.btn_avg      = self._op_btn(c, "Average Filter",  lambda: self.on_spatial_filter("average"))
        self.btn_gaussian = self._op_btn(c, "Gaussian Filter", lambda: self.on_spatial_filter("gaussian"))
        self.btn_median   = self._op_btn(c, "Median Filter",   self.on_median_filter)
        self._muted_label(c, "Edge Detection")
        self.btn_sobel    = self._op_btn(c, "Sobel Edge",   lambda: self.on_edge_detection("sobel"))
        self.btn_prewitt  = self._op_btn(c, "Prewitt Edge", lambda: self.on_edge_detection("prewitt"))

        # Contrast
        c = self._make_card("Contrast Enhancement")
        self._muted_label(c, "Block Size  (pixels)")
        self.block_size_input = self._styled_entry(c, placeholder="e.g. 32")
        self.btn_hist_eq = self._op_btn(c, "Apply Local Equalization", self.on_local_hist_eq)

        # Morphology
        c = self._make_card("Morphology")
        self._muted_label(c, "Structuring Element  Shape")
        self.se_shape_var = ctk.StringVar(value="Square")
        self.se_dropdown  = self._styled_optionmenu(c, ["Square", "Cross"], self.se_shape_var)
        self._muted_label(c, "SE Size  (odd integer)")
        self.se_size_input = self._styled_entry(c, default="3")
        self._muted_label(c, "Operations")
        self.btn_erode    = self._op_btn(c, "Erosion",             lambda: self.on_morphology("erode"))
        self.btn_dilate   = self._op_btn(c, "Dilation",            lambda: self.on_morphology("dilate"))
        self.btn_opening  = self._op_btn(c, "Opening",             lambda: self.on_morphology("open"))
        self.btn_closing  = self._op_btn(c, "Closing",             lambda: self.on_morphology("close"))
        self.btn_boundary = self._op_btn(c, "Boundary Extraction", lambda: self.on_morphology("boundary"))

        # Pipeline Controls
        c = self._make_card("Pipeline Controls")
        self.btn_undo  = self._op_btn(c, "⟵  Undo Last Step",   self.on_undo,  fg=CLR_WARNING, hover=CLR_WARNING_HVR)
        self.btn_reset = self._op_btn(c, "↺  Reset to Original", self.on_reset, fg=CLR_DANGER,  hover=CLR_DANGER_HVR)

    # ─────────────────────────────────────────────────────────────────────────
    # Status helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, msg, level="info"):
        palette = {
            "info":    (CLR_ST_IDLE,    CLR_TEXT_SEC),
            "success": (CLR_ST_SUCCESS, "#3fb950"),
            "warning": (CLR_ST_WARNING, "#d29922"),
            "error":   (CLR_ST_ERROR,   "#f85149"),
            "busy":    (CLR_ST_BUSY,    CLR_TEXT_HDG),
        }
        bg, fg = palette.get(level, palette["info"])
        self.status_bar.configure(fg_color=bg)
        self.status_label.configure(text=f"  {msg}", text_color=fg)
        self.master.update_idletasks()

    def _update_dim_label(self):
        if self.current_image_array is not None:
            h, w = self.current_image_array.shape[:2]
            mode = "RGB" if self.current_image_array.ndim == 3 else "Grayscale"
            steps = len(self.image_history)
            self.dim_label.configure(text=f"{w} × {h}  {mode}  |  History: {steps}  ")
            self.history_label.configure(text=f"Undo steps: {steps}")
        else:
            self.dim_label.configure(text="")

    def _update_stats(self):
        if self.current_image_array is None:
            return
        arr = self.current_image_array
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        self.stats_label.configure(
            text=(
                f"Size:     {w} × {h} px\n"
                f"Channels: {ch}\n"
                f"Dtype:    {arr.dtype}\n"
                f"Min/Max:  {int(arr.min())} / {int(arr.max())}\n"
                f"Mean:     {arr.mean():.1f}"
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Input validation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_positive_float(self, widget, name):
        try:
            val = float(widget.get().strip())
        except ValueError:
            self._set_status(f"{name}: enter a valid number.", "error")
            return None
        if val <= 0:
            self._set_status(f"{name}: must be positive.", "error")
            return None
        return val

    def _parse_odd_int(self, widget, name):
        try:
            val = int(widget.get().strip())
        except ValueError:
            self._set_status(f"{name}: must be an odd integer (e.g. 3, 5, 7).", "error")
            return None
        if val <= 0:
            self._set_status(f"{name}: must be a positive odd integer.", "error")
            return None
        if val % 2 == 0:
            self._set_status(f"{name}: must be odd (1, 3, 5, 7 …).", "error")
            return None
        return val

    def _parse_positive_int(self, widget, name):
        try:
            val = int(widget.get().strip())
        except ValueError:
            self._set_status(f"{name}: must be a positive integer.", "error")
            return None
        if val <= 0:
            self._set_status(f"{name}: must be positive.", "error")
            return None
        return val

    def _require_image(self):
        if self.current_image_array is None:
            self._set_status("No image loaded — use  Load Image  first.", "warning")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Core display / state helpers
    # ─────────────────────────────────────────────────────────────────────────

    def show_error(self, message):
        self._set_status(message[:90], "error")
        messagebox.showerror("Error", message)

    def show_info(self, message):
        self._set_status(message[:90], "success")
        messagebox.showinfo("Success", message)

    def refresh_canvas(self, new_numpy_array):
        if new_numpy_array is None:
            return

        if new_numpy_array.dtype != np.uint8:
            mn, mx = new_numpy_array.min(), new_numpy_array.max()
            if mx > mn:
                norm = ((new_numpy_array - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(new_numpy_array, dtype=np.uint8)
        else:
            norm = new_numpy_array

        pil_img = Image.fromarray(norm)
        w, h = pil_img.size
        # Scale down for display only; original array is unchanged
        scale   = min(1.0, _MAX_DISPLAY_PX / max(w, h, 1))
        disp_w  = max(1, int(w * scale))
        disp_h  = max(1, int(h * scale))

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))
        self.image_label.configure(image=ctk_img, text="")
        self._current_tk_image = ctk_img   # prevent GC

        self._update_dim_label()
        self._update_stats()

    def update_metadata_panel(self, metadata_dict):
        self.metadata_textbox.configure(state="normal")
        self.metadata_textbox.delete("1.0", "end")
        if not metadata_dict:
            self.metadata_textbox.insert("end", "No metadata available.")
        else:
            for key, val in metadata_dict.items():
                self.metadata_textbox.insert("end", f"{key}:\n  {val}\n\n")
        self.metadata_textbox.configure(state="disabled")

    def push_state(self):
        if self.current_image_array is not None:
            self.image_history.append(self.current_image_array.copy())

    def apply_action(self, action_func, *args, **kwargs):
        """Wrapper: push state, run action, refresh display. Returns True on success."""
        if self.current_image_array is None:
            self._set_status("No image loaded.", "warning")
            return False
        try:
            self.push_state()
            new_image = action_func(self.current_image_array, *args, **kwargs)
            if new_image is None:
                if self.image_history:
                    self.image_history.pop()
                self._set_status("This operation is not yet implemented.", "warning")
                return False
            self.current_image_array = new_image
            h, w = self.current_image_array.shape[:2]
            self.metadata_dict["Width"]  = str(w)
            self.metadata_dict["Height"] = str(h)
            self.update_metadata_panel(self.metadata_dict)
            self.refresh_canvas(self.current_image_array)
            return True
        except Exception as e:
            if self.image_history:
                self.image_history.pop()
            self._set_status(f"Error: {str(e)[:70]}", "error")
            messagebox.showerror("Engine Error", str(e))
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks  (names unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def on_load_image(self):
        path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Medical Images", "*.png *.jpg *.jpeg *.bmp *.dcm *.dicom"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self._set_status("Loading image …", "busy")
        try:
            image_array, metadata, err = image_io.load_image(path)
            if err:
                self.show_error(err)
                return
            self.original_image_array = image_array.copy()
            self.current_image_array  = image_array.copy()
            self.image_history.clear()
            self.metadata_dict = metadata
            self.refresh_canvas(self.current_image_array)
            self.update_metadata_panel(self.metadata_dict)
            self._set_status("Image loaded successfully.", "success")
        except Exception as e:
            self.show_error(f"Error loading image:\n{e}")

    def on_save_image(self):
        if not self._require_image():
            return
        path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        self._set_status("Saving …", "busy")
        try:
            success, msg = image_io.save_image(self.current_image_array, path)
            if success:
                self._set_status("Image saved.", "success")
                messagebox.showinfo("Saved", msg)
            else:
                self.show_error(msg)
        except Exception as e:
            self.show_error(f"Error saving image:\n{e}")

    def on_zoom(self):
        if not self._require_image():
            return
        scale = self._parse_positive_float(self.scale_input, "Scale factor")
        if scale is None:
            return
        if scale > 8.0:
            self._set_status(f"Warning: scale {scale}× is very large — may be slow.", "warning")
        interp = self.interp_type_var.get()
        self._set_status(f"Applying {interp} zoom ×{scale:.2f} …", "busy")
        if interp == "Nearest Neighbor":
            ok = self.apply_action(spatial_engine.zoom_nearest_neighbor, scale)
        else:
            ok = self.apply_action(spatial_engine.zoom_linear, scale)
        if ok:
            self._set_status(f"Zoom ×{scale:.2f} applied ({interp}).", "success")

    def on_spatial_filter(self, filter_type):
        if not self._require_image():
            return
        k = self._parse_odd_int(self.kernel_size_input, "Kernel size")
        if k is None:
            return
        var = self.variance_slider.get()
        self._set_status(f"Applying {filter_type.capitalize()} filter (k={k}) …", "busy")
        ok = self.apply_action(spatial_engine.apply_smoothing_filter, filter_type, k, variance=var)
        if ok:
            self._set_status(f"{filter_type.capitalize()} filter applied (k={k}).", "success")

    def on_median_filter(self):
        if not self._require_image():
            return
        k = self._parse_odd_int(self.kernel_size_input, "Kernel size")
        if k is None:
            return
        self._set_status(f"Applying Median filter (k={k}) …", "busy")
        ok = self.apply_action(spatial_engine.apply_median_filter, k)
        if ok:
            self._set_status(f"Median filter applied (k={k}).", "success")

    def on_edge_detection(self, operator_type):
        if not self._require_image():
            return
        self._set_status(f"Applying {operator_type.capitalize()} edge detection …", "busy")
        ok = self.apply_action(spatial_engine.apply_edge_detection, operator_type)
        if ok:
            self._set_status(f"{operator_type.capitalize()} edge detection applied.", "success")

    def on_local_hist_eq(self):
        if not self._require_image():
            return
        b = self._parse_positive_int(self.block_size_input, "Block size")
        if b is None:
            return
        self._set_status(f"Applying local histogram equalization (block={b}) …", "busy")
        ok = self.apply_action(spatial_engine.local_histogram_equalization, b)
        if ok:
            self._set_status(f"Local equalization applied (block={b} px).", "success")

    def on_morphology(self, operation):
        if not self._require_image():
            return
        size = self._parse_odd_int(self.se_size_input, "SE size")
        if size is None:
            return
        shape = self.se_shape_var.get()
        op_labels = {
            "erode":    "Erosion",
            "dilate":   "Dilation",
            "open":     "Opening",
            "close":    "Closing",
            "boundary": "Boundary Extraction",
        }
        label = op_labels.get(operation, operation.capitalize())
        try:
            se = morphology_engine.generate_structuring_element(shape, size)
        except Exception as e:
            self._set_status(f"SE error: {e}", "error")
            return
        func_map = {
            "erode":    morphology_engine.erode,
            "dilate":   morphology_engine.dilate,
            "open":     morphology_engine.opening,
            "close":    morphology_engine.closing,
            "boundary": morphology_engine.extract_boundary,
        }
        func = func_map.get(operation)
        if func is None:
            self._set_status("Unknown morphology operation.", "error")
            return
        self._set_status(f"Applying {label} ({shape} SE, {size}×{size}) …", "busy")
        ok = self.apply_action(func, se)
        if ok:
            self._set_status(f"{label} applied ({shape} SE, size={size}).", "success")

    def on_undo(self):
        if not self.image_history:
            self._set_status("Nothing to undo.", "warning")
            return
        self.current_image_array = self.image_history.pop()
        self.refresh_canvas(self.current_image_array)
        remaining = len(self.image_history)
        self._set_status(f"Undo complete. {remaining} step(s) remaining.", "info")

    def on_reset(self):
        if self.original_image_array is None:
            self._set_status("No original image to restore.", "warning")
            return
        self.push_state()
        self.current_image_array = self.original_image_array.copy()
        h, w = self.current_image_array.shape[:2]
        self.metadata_dict["Width"]  = str(w)
        self.metadata_dict["Height"] = str(h)
        self.refresh_canvas(self.current_image_array)
        self.update_metadata_panel(self.metadata_dict)
        self._set_status("Reset to original image.", "info")
