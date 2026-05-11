# Author: Zeyad Khaled (System Architect & Core Engine) - Complete Ownership
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import numpy as np
from PIL import Image, ImageTk

import ai_segmentation_engine
import image_io
import spatial_engine
import morphology_engine
import frequency_engine

from config import (
    UI_PADDING, CORNER_RADIUS, SIDEBAR_WIDTH,
    MAX_PROCESSING_PIXELS, MAX_ZOOM_SCALE,
    AI_VIDEO_FRAME_STEP, AI_VIDEO_MAX_FRAMES,
    CLR_BG_MAIN, CLR_BG_SIDEBAR, CLR_BG_CARD, CLR_BG_CARD_HDR, CLR_BORDER,
    CLR_AI_ACCENT, CLR_AI_PANEL, CLR_AI_CARD,
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

# ─── State model ──────────────────────────────────────────────────────────────
#  State Management & Undo Logic:
#  - image_history  = [loaded_copy, result1, result2, ...]
#    This acts as our pipeline stack. Every time an engine returns a new array, 
#    we append a safe .copy() here.
#  - current_image  = image_history[-1]   (always)
#    The UI engines always pull the active working image from the top of the stack.
#  - original_image = never mutated after load. Safe anchor for full resets.
#
#  Undo:  pop history tail → current = new tail  (only if len > 1)
#         This reverts the pipeline to the exact state before the last operation.
#  Reset: history = [original_copy]  → current = original_copy
# ──────────────────────────────────────────────────────────────────────────────


class UIManager:
    def __init__(self, master):
        self.master = master
        self.master.minsize(1100, 700)

        self.original_image_array = None
        self.current_image_array  = None
        self.image_history: list  = []    # always contains copies; never None
        self.redo_history: list   = []
        self.metadata_dict        = {}
        # Bahr-Phase2: ROI selection state variables
        self.selected_roi         = None
        self.last_roi_histogram   = None
        # Youssra-Phase2
        self.selection_mode        = "roi"
        self.selected_template_roi = None
        self.selected_template     = None
        self.last_template_match   = None

        # Owner: Bahr - AI Segmentation Bonus
        # AI state is intentionally separate from current_image_array/image_history.
        self.ai_model             = None
        # AI-Segmentation
        self.ai_model_path        = "ai_models/polyp_model.keras"
        self.ai_image             = None
        self.ai_mask              = None
        self.ai_overlay           = None
        self.ai_confidence        = None
        self.ai_display_mode      = None
        # AI-Segmentation START: auto-load and RGB fix
        self.ai_input_source      = None
        self.ai_image_mode        = None
        # AI-Segmentation END: auto-load and RGB fix
        # AI-Segmentation START: robust upload/analyze/clear fix
        self.ai_rgb_file_path     = None   # path of last AI-only RGB upload
        self.ai_image_was_grayscale = False  # flagged when pipeline gray→RGB copy used
        # AI-Segmentation END: robust upload/analyze/clear fix
        self.ai_video_path        = None
        self.ai_video_results     = []

        # Canvas display references (GC anchors)
        self._photo_image          = None
        self._canvas_image_id      = None
        self._canvas_placeholder_id = None
        # Bahr-Phase2: ROI drawing state and display mapping state
        self._roi_rect_id           = None
        self._roi_drag_start        = None
        # Youssra-Phase2
        self._template_rect_id      = None
        self._template_drag_start   = None
        self._match_box_id          = None
        self._display_image_origin  = (0, 0)
        self._display_image_size    = (0, 0)
        self._display_to_array_scale = (1.0, 1.0)
        self._ai_preview_photos     = {}
        # AI-Segmentation START: PhotoImage reference fix
        self.ai_original_photo      = None
        self.ai_overlay_photo       = None
        self.ai_mask_photo          = None
        # AI-Segmentation START: complete image reference audit
        self.ai_display_photo       = None
        self.ai_original_ctk_image  = None
        self.ai_overlay_ctk_image   = None
        self.ai_mask_ctk_image      = None
        self.ai_original_canvas_item = None
        self.ai_overlay_canvas_item = None
        self.ai_mask_canvas_item    = None
        # AI-Segmentation END: complete image reference audit
        self.ai_original_image_id   = None
        self.ai_overlay_image_id    = None
        self.ai_mask_image_id       = None
        # AI-Segmentation END: PhotoImage reference fix

        self.setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    # Author: Zeyad Khaled
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
        self._bind_keyboard_shortcuts()

    # Author: Zeyad Khaled
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

    # Author: Zeyad Khaled
    def _build_viewer(self):
        """
        Viewer = CTkFrame  (column 1, fills all available space)
          ├── tk.Canvas    (row 0, col 0, expandable) — shows image at 1:1 pixels
          ├── v_scroll     (row 0, col 1, fills ns)
          └── h_scroll     (row 1, col 0, fills ew)
        """
        self.viewer_frame = ctk.CTkFrame(
            self.master, corner_radius=0, fg_color=CLR_BG_MAIN,
        )
        self.viewer_frame.grid(row=0, column=1, sticky="nsew")
        self.viewer_frame.grid_rowconfigure(0, weight=1)
        self.viewer_frame.grid_columnconfigure(0, weight=1)

        self.main_tabs = ctk.CTkTabview(
            self.viewer_frame,
            fg_color=CLR_BG_MAIN,
            segmented_button_fg_color=CLR_BG_CARD_HDR,
            segmented_button_selected_color=CLR_ACCENT,
            segmented_button_selected_hover_color=CLR_ACCENT_HOVER,
            segmented_button_unselected_color=CLR_BG_CARD,
            segmented_button_unselected_hover_color=CLR_ACCENT_DIM,
        )
        self.main_tabs.grid(row=0, column=0, sticky="nsew")
        self.pipeline_tab = self.main_tabs.add("Required Pipeline")
        self.ai_tab = self.main_tabs.add("Polyp AI Detection")
        self.pipeline_tab.configure(fg_color=CLR_BG_MAIN)
        self.ai_tab.configure(fg_color=CLR_AI_PANEL)
        self.pipeline_tab.grid_rowconfigure(0, weight=1)
        self.pipeline_tab.grid_rowconfigure(1, weight=0)
        self.pipeline_tab.grid_columnconfigure(0, weight=1)
        self.pipeline_tab.grid_columnconfigure(1, weight=0)

        # Main canvas
        self.canvas = tk.Canvas(
            self.pipeline_tab,
            bg=CLR_BG_MAIN,
            highlightthickness=0,
            bd=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        self.v_scroll = tk.Scrollbar(
            self.pipeline_tab,
            orient="vertical",
            command=self.canvas.yview,
            bg=CLR_BG_CARD_HDR,
            troughcolor=CLR_BG_CARD,
            activebackground=CLR_ACCENT,
            width=12,
            relief="flat",
            borderwidth=0,
        )
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = tk.Scrollbar(
            self.pipeline_tab,
            orient="horizontal",
            command=self.canvas.xview,
            bg=CLR_BG_CARD_HDR,
            troughcolor=CLR_BG_CARD,
            activebackground=CLR_ACCENT,
            width=12,
            relief="flat",
            borderwidth=0,
        )
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )

        # Centered placeholder text (updated on resize)
        self._canvas_placeholder_id = self.canvas.create_text(
            600, 300,
            text="No image loaded\n\nPress  Load Image  in the sidebar to begin",
            fill=CLR_TEXT_SEC,
            font=("Segoe UI", 14),
            justify="center",
        )

        # Mouse / keyboard bindings
        self.canvas.bind("<Configure>",         self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>",         self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>",   self._on_shift_mousewheel)
        self.canvas.bind("<ButtonPress-1>",      self._on_roi_start)
        self.canvas.bind("<B1-Motion>",          self._on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>",    self._on_roi_end)
        self.canvas.bind("<ButtonPress-2>",      self._on_pan_start)
        self.canvas.bind("<B2-Motion>",          self._on_pan_move)
        self.canvas.bind("<Enter>",              lambda e: self.canvas.focus_set())
        self._build_ai_tab()

    # Owner: Zeyad - GUI / Pipeline Architecture
    def _build_ai_tab(self):
        """Build the isolated polyp AI workstation tab."""
        # AI-Segmentation START: analyze rendering fix
        # Row 0 = header (fixed), row 1 = content panels (expandable), row 2 = bottom bar (fixed).
        # Explicit weight=0 on rows 0 and 2 prevents accidental expansion that could
        # push controls into the preview columns.
        self.ai_tab.grid_rowconfigure(0, weight=0)
        self.ai_tab.grid_rowconfigure(1, weight=1)
        self.ai_tab.grid_rowconfigure(2, weight=0)
        # Column 0 = AI controls (fixed width), columns 1-2 = preview panels (expand equally).
        self.ai_tab.grid_columnconfigure(0, weight=0)
        self.ai_tab.grid_columnconfigure(1, weight=1)
        self.ai_tab.grid_columnconfigure(2, weight=1)
        # AI-Segmentation END: analyze rendering fix

        header = ctk.CTkFrame(self.ai_tab, fg_color=CLR_AI_PANEL, corner_radius=0)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            header,
            text="Polyp AI Detection / Segmentation",
            font=("Segoe UI", 20, "bold"),
            text_color=CLR_AI_ACCENT,
            anchor="w",
        ).pack(side="left", padx=4)
        self.ai_status_label = ctk.CTkLabel(
            header,
            text="Ready",
            font=FONT_LABEL_BOLD,
            text_color=CLR_TEXT_SEC,
            anchor="e",
        )
        self.ai_status_label.pack(side="right", padx=4)

        # AI-Segmentation START: UX simplification
        # AI-Segmentation START: analyze rendering fix
        # All AI controls live exclusively inside self.ai_controls_frame (column 0).
        # Preview panels are separate frames in columns 1-2.  No widget is ever
        # packed/gridded into a preview frame from this controls section.
        self.ai_controls_frame = ctk.CTkScrollableFrame(
            self.ai_tab,
            width=270,
            fg_color=CLR_AI_CARD,
            border_color=CLR_BORDER,
            border_width=1,
            corner_radius=8,
        )
        controls = self.ai_controls_frame  # local alias kept for readability below
        controls.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(0, 12))
        # AI-Segmentation END: analyze rendering fix
        ctk.CTkLabel(
            controls,
            text="AI Controls",
            font=FONT_SECTION_HDG,
            text_color=CLR_AI_ACCENT,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))

        self.ai_info_label = ctk.CTkLabel(
            controls,
            text="Model loaded: No\nAI image ready: No\nMask ready: No\nConfidence: --",
            font=FONT_MONO,
            text_color=CLR_TEXT_SEC,
            wraplength=235,
            justify="left",
            anchor="w",
        )
        self.ai_info_label.pack(fill="x", padx=10, pady=(0, 8))

        self._op_btn(
            controls,
            "Analyze Image",
            self.on_ai_run_current_image,
            fg=CLR_AI_ACCENT,
            hover=CLR_SUCCESS_HVR,
        )
        self._ai_help_text(
            controls,
            "Loads the model if needed, uses the uploaded AI image if present, otherwise copies the current pipeline image.",
        )
        # AI-Segmentation START: auto-load and RGB fix
        self._op_btn(controls, "Upload AI Image (RGB)", self.on_ai_upload_image, fg=CLR_CYAN, hover=CLR_CYAN_HOVER)
        self._ai_help_text(controls, "Loads an RGB image directly for AI without using the normal pipeline loader.")
        # AI-Segmentation END: auto-load and RGB fix

        cfg = ai_segmentation_engine.load_ai_config()
        self.ai_threshold_var = ctk.DoubleVar(value=float(cfg.get("threshold", 0.5)))
        self._muted_label(controls, "AI Threshold")
        self.ai_threshold_slider = ctk.CTkSlider(
            controls,
            from_=0.1,
            to=0.9,
            number_of_steps=80,
            variable=self.ai_threshold_var,
            progress_color=CLR_AI_ACCENT,
            button_color=CLR_TEXT_HDG,
            button_hover_color=CLR_TEXT_PRI,
            command=lambda value: self.ai_threshold_label.configure(text=f"Threshold: {float(value):.2f}"),
        )
        self.ai_threshold_slider.pack(fill="x", padx=10, pady=(2, 0))
        self.ai_threshold_label = ctk.CTkLabel(
            controls,
            text=f"Threshold: {self.ai_threshold_var.get():.2f}",
            font=FONT_LABEL,
            text_color=CLR_TEXT_SEC,
        )
        self.ai_threshold_label.pack(fill="x", padx=10, pady=(0, 6))
        self._ai_help_text(
            controls,
            "Threshold controls how strict the mask is. Higher = smaller/stricter mask, lower = larger/more sensitive mask.",
        )

        self._op_btn(controls, "Toggle View: Overlay / Mask", self.on_ai_toggle_view)
        self._ai_help_text(controls, "Switches between green overlay and black-white mask.")
        self._op_btn(controls, "Save AI Result", self.on_ai_save_result, fg=CLR_WARNING, hover=CLR_WARNING_HVR)
        self._ai_help_text(controls, "Saves the displayed AI mask or overlay.")
        self._op_btn(controls, "Clear AI Workspace", self.on_ai_clear_workspace, fg=CLR_DANGER, hover=CLR_DANGER_HVR)
        self._ai_help_text(controls, "Clears AI-only image, mask, overlay, and confidence.")
        # AI-Segmentation END: UX simplification

        # AI-Segmentation START: analyze rendering fix
        # Each preview panel is stored as a named instance attribute so layout can be
        # audited at runtime.  Only the corresponding preview label is ever packed inside.
        self.ai_original_preview_frame = self._make_ai_panel(self.ai_tab, "Original Image / Frame")
        self.ai_original_preview_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=(0, 12))
        # AI-Segmentation START: robust upload/analyze/clear fix
        # tk.Canvas widgets — see _make_ai_preview_canvas for why canvas instead of CTkLabel.
        self.ai_original_preview = self._make_ai_preview_canvas(self.ai_original_preview_frame, "No AI image loaded")
        # AI-Segmentation END: robust upload/analyze/clear fix

        self.ai_overlay_preview_frame = self._make_ai_panel(self.ai_tab, "Segmentation Overlay")
        self.ai_overlay_preview_frame.grid(row=1, column=2, sticky="nsew", padx=(6, 12), pady=(0, 12))
        # AI-Segmentation START: robust upload/analyze/clear fix
        self.ai_overlay_preview = self._make_ai_preview_canvas(self.ai_overlay_preview_frame, "Overlay preview")
        # AI-Segmentation END: robust upload/analyze/clear fix

        bottom = ctk.CTkFrame(self.ai_tab, fg_color=CLR_AI_PANEL, corner_radius=0)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=2)

        self.ai_mask_preview_frame = self._make_ai_panel(bottom, "Mask Preview")
        self.ai_mask_preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        # AI-Segmentation START: robust upload/analyze/clear fix
        self.ai_mask_preview = self._make_ai_preview_canvas(self.ai_mask_preview_frame, "Mask preview", height=170)
        # AI-Segmentation END: robust upload/analyze/clear fix
        # AI-Segmentation END: analyze rendering fix

        status_panel = self._make_ai_panel(bottom, "Diagnosis / Status")
        status_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.ai_diagnosis_text = ctk.CTkTextbox(
            status_panel,
            height=150,
            font=FONT_MONO,
            fg_color=CLR_BG_MAIN,
            text_color=CLR_TEXT_PRI,
            corner_radius=8,
            wrap="word",
        )
        self.ai_diagnosis_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.ai_diagnosis_text.insert(
            "end",
            "Ready\nUpload an RGB image for AI or use the current pipeline image, then click Analyze Image. "
            "The model loads automatically when needed.",
        )
        self.ai_diagnosis_text.configure(state="disabled")

    def _make_ai_panel(self, parent, title):
        panel = ctk.CTkFrame(parent, fg_color=CLR_AI_CARD, border_color=CLR_BORDER, border_width=1, corner_radius=8)
        ctk.CTkLabel(
            panel,
            text=title,
            font=FONT_SECTION_HDG,
            text_color=CLR_AI_ACCENT,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 0))
        return panel

    # AI-Segmentation START: robust upload/analyze/clear fix
    def _make_ai_preview_canvas(self, parent, placeholder_text, height=360):
        """Create a tk.Canvas preview widget.

        Using plain tk.Canvas instead of CTkLabel eliminates the 'pyimageX doesn't
        exist' error class entirely:
          - canvas.delete("all") removes the Tk-level image reference synchronously,
            before we release any Python-side PhotoImage reference.
          - There is no deferred _draw() or internal _image state to race against GC.
          - canvas._ai_photo is the single persistent reference anchor on the widget.
        """
        canvas = tk.Canvas(
            parent,
            bg=CLR_BG_MAIN,
            highlightthickness=0,
            bd=0,
            height=height,
        )
        canvas.pack(fill="both", expand=True, padx=10, pady=10)
        canvas._ai_photo = None
        canvas._ai_placeholder = placeholder_text
        # Draw placeholder once the widget is laid out and has real dimensions.
        canvas.bind("<Configure>", lambda e, c=canvas: self._ai_canvas_redraw_placeholder(c))
        return canvas

    def _ai_canvas_redraw_placeholder(self, canvas):
        """Re-center placeholder text when the canvas is resized (only if no image)."""
        if canvas._ai_photo is not None:
            return  # image is showing — don't overwrite with placeholder
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        canvas.delete("all")
        canvas.create_text(
            w // 2, h // 2,
            text=canvas._ai_placeholder,
            fill=CLR_TEXT_SEC,
            font=FONT_LABEL,
            justify="center",
        )
    # AI-Segmentation END: robust upload/analyze/clear fix

    def _make_ai_preview_label(self, parent, text, height=360):
        """Legacy shim — kept so nothing else breaks; routes to canvas version."""
        return self._make_ai_preview_canvas(parent, text, height)

    # AI-Segmentation START: UX simplification
    def _ai_help_text(self, parent, text):
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=FONT_LABEL,
            text_color=CLR_TEXT_SEC,
            wraplength=235,
            justify="left",
            anchor="w",
        )
        label.pack(fill="x", padx=10, pady=(0, 6))
        return label
    # AI-Segmentation END: UX simplification

    # Author: Zeyad Khaled
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

        roi_card = ctk.CTkFrame(self.right_panel, fg_color=CLR_BG_CARD, corner_radius=8)
        roi_card.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(
            roi_card,
            text="ROI Statistics",
            font=FONT_LABEL_BOLD,
            text_color=CLR_TEXT_HDG,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 0))
        self.roi_stats_label = ctk.CTkLabel(
            roi_card,
            text="Draw a rectangle on the image.",
            font=FONT_MONO,
            text_color=CLR_TEXT_SEC,
            anchor="w",
            justify="left",
        )
        self.roi_stats_label.pack(fill="x", padx=10, pady=8)
        # Bahr-Phase2 START: ROI histogram display
        self.roi_hist_canvas = tk.Canvas(
            roi_card,
            height=96,
            bg=CLR_BG_MAIN,
            highlightthickness=1,
            highlightbackground=CLR_BORDER,
            bd=0,
        )
        self.roi_hist_canvas.pack(fill="x", padx=10, pady=(0, 10))
        self._draw_roi_histogram(None)
        # Bahr-Phase2 END: ROI histogram display

        # Youssra-Phase2 START: template matching UI
        template_card = ctk.CTkFrame(self.right_panel, fg_color=CLR_BG_CARD, corner_radius=8)
        template_card.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(
            template_card,
            text="Template Match",
            font=FONT_LABEL_BOLD,
            text_color=CLR_TEXT_HDG,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 0))
        self.template_match_label = ctk.CTkLabel(
            template_card,
            text="Mode: ROI statistics",
            font=FONT_MONO,
            text_color=CLR_TEXT_SEC,
            anchor="w",
            justify="left",
        )
        self.template_match_label.pack(fill="x", padx=10, pady=8)
        # Youssra-Phase2 END: template matching UI

        pipeline_card = ctk.CTkFrame(self.right_panel, fg_color=CLR_BG_CARD, corner_radius=8)
        pipeline_card.pack(side="bottom", fill="x", padx=8, pady=(4, 8))
        ctk.CTkLabel(
            pipeline_card,
            text="Pipeline",
            font=FONT_LABEL_BOLD,
            text_color=CLR_TEXT_HDG,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 2))
        pipe_row = ctk.CTkFrame(pipeline_card, fg_color="transparent")
        pipe_row.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_undo = ctk.CTkButton(
            pipe_row, text="Undo", command=self.on_undo,
            font=FONT_BUTTON, fg_color=CLR_WARNING, hover_color=CLR_WARNING_HVR,
            corner_radius=6, height=32, width=70,
        )
        self.btn_undo.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_redo = ctk.CTkButton(
            pipe_row, text="Redo", command=self.on_redo,
            font=FONT_BUTTON, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_HOVER,
            corner_radius=6, height=32, width=70,
        )
        self.btn_redo.pack(side="left", fill="x", expand=True, padx=3)
        self.btn_reset = ctk.CTkButton(
            pipe_row, text="Reset", command=self.on_reset,
            font=FONT_BUTTON, fg_color=CLR_DANGER, hover_color=CLR_DANGER_HVR,
            corner_radius=6, height=32, width=70,
        )
        self.btn_reset.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.history_label = ctk.CTkLabel(
            pipeline_card,
            text="Undo: 0 | Redo: 0",
            font=FONT_LABEL,
            text_color=CLR_TEXT_SEC,
        )
        self.history_label.pack(fill="x", padx=10, pady=(0, 8))

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
    # Canvas event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        """Keep placeholder text centered when the window is resized."""
        if self._canvas_image_id is None and self._canvas_placeholder_id is not None:
            self.canvas.coords(
                self._canvas_placeholder_id,
                event.width // 2,
                event.height // 2,
            )

    def _on_mousewheel(self, event):
        """Vertical scroll via mouse wheel (Windows: event.delta = ±120)."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        """Horizontal scroll via Shift + mouse wheel."""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    # Bahr-Phase2 START: ROI coordinate mapping fix
    def _canvas_to_image_xy(self, event):
        """
        Map a mouse event on the scrollable canvas to real image-array pixels.

        The required pipeline currently renders the image at original size with
        its top-left corner at canvas coordinate (0, 0). This helper still tracks
        display origin, display size, and display-to-array scale so ROI selection
        remains correct if a later UI change centers or scales the canvas image.
        """
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        origin_x, origin_y = self._display_image_origin
        scale_x, scale_y = self._display_to_array_scale

        image_x = int(round((canvas_x - origin_x) * scale_x))
        image_y = int(round((canvas_y - origin_y) * scale_y))

        if self.current_image_array is None:
            return image_x, image_y

        height, width = self.current_image_array.shape[:2]
        image_x = max(0, min(image_x, width))
        image_y = max(0, min(image_y, height))
        return image_x, image_y

    def _image_to_canvas_xy(self, x, y):
        """Map stored image-array ROI coordinates back to displayed canvas coordinates."""
        origin_x, origin_y = self._display_image_origin
        scale_x, scale_y = self._display_to_array_scale
        display_x = origin_x + (x / scale_x if scale_x else x)
        display_y = origin_y + (y / scale_y if scale_y else y)
        return display_x, display_y
    # Bahr-Phase2 END: ROI coordinate mapping fix

    # Bahr-Phase2
    def _on_roi_start(self, event):
        """Begin drawing a rectangular ROI on the image canvas."""
        if self.current_image_array is None:
            return
        if self.selection_mode == "template":
            self._on_template_start(event)
            return
        x, y = self._canvas_to_image_xy(event)
        self._roi_drag_start = (x, y)
        self.selected_roi = None
        self.last_roi_histogram = None
        self._draw_roi_histogram(None)
        if self._roi_rect_id is not None:
            self.canvas.delete(self._roi_rect_id)
        cx, cy = self._image_to_canvas_xy(x, y)
        self._roi_rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline=CLR_CYAN, width=2, dash=(4, 2),
        )

    # Bahr-Phase2
    def _on_roi_drag(self, event):
        """Update the visible ROI rectangle while dragging."""
        if self.current_image_array is None or self._roi_drag_start is None:
            if self.selection_mode == "template" and self._template_drag_start is not None:
                self._on_template_drag(event)
            return
        if self.selection_mode == "template":
            self._on_template_drag(event)
            return
        x1, y1 = self._roi_drag_start
        x2, y2 = self._canvas_to_image_xy(event)
        cx1, cy1 = self._image_to_canvas_xy(x1, y1)
        cx2, cy2 = self._image_to_canvas_xy(x2, y2)
        if self._roi_rect_id is None:
            self._roi_rect_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline=CLR_CYAN, width=2, dash=(4, 2),
            )
        else:
            self.canvas.coords(self._roi_rect_id, cx1, cy1, cx2, cy2)

    # Bahr-Phase2
    def _on_roi_end(self, event):
        """Finish ROI selection and store clamped image coordinates."""
        if self.current_image_array is None or self._roi_drag_start is None:
            if self.selection_mode == "template" and self._template_drag_start is not None:
                self._on_template_end(event)
            return
        if self.selection_mode == "template":
            self._on_template_end(event)
            return
        x1, y1 = self._roi_drag_start
        x2, y2 = self._canvas_to_image_xy(event)
        self._roi_drag_start = None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right <= left or bottom <= top:
            self.clear_roi(show_message=False)
            self._set_status("Please select a valid ROI.", "warning")
            return

        self.selected_roi = (left, top, right, bottom)
        if self._roi_rect_id is not None:
            cx1, cy1 = self._image_to_canvas_xy(left, top)
            cx2, cy2 = self._image_to_canvas_xy(right, bottom)
            self.canvas.coords(self._roi_rect_id, cx1, cy1, cx2, cy2)
        self.roi_stats_label.configure(
            text=f"Selected ROI:\nx1={left}, y1={top}\nx2={right}, y2={bottom}"
        )
        self._set_status("ROI selected.", "info")

    # Youssra-Phase2 START: template matching UI
    def _on_template_start(self, event):
        """Begin drawing a template selection rectangle without touching Bahr ROI state."""
        x, y = self._canvas_to_image_xy(event)
        self._template_drag_start = (x, y)
        self.selected_template_roi = None
        if self._template_rect_id is not None:
            self.canvas.delete(self._template_rect_id)
        cx, cy = self._image_to_canvas_xy(x, y)
        self._template_rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline=CLR_SUCCESS, width=2,
        )
        self.template_match_label.configure(text="Template mode:\ndrag a template box")

    def _on_template_drag(self, event):
        """Update the template selection rectangle."""
        if self._template_drag_start is None:
            return
        x1, y1 = self._template_drag_start
        x2, y2 = self._canvas_to_image_xy(event)
        cx1, cy1 = self._image_to_canvas_xy(x1, y1)
        cx2, cy2 = self._image_to_canvas_xy(x2, y2)
        if self._template_rect_id is None:
            self._template_rect_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline=CLR_SUCCESS, width=2,
            )
        else:
            self.canvas.coords(self._template_rect_id, cx1, cy1, cx2, cy2)

    def _on_template_end(self, event):
        """Store the selected template rectangle in image coordinates."""
        if self._template_drag_start is None:
            return
        x1, y1 = self._template_drag_start
        x2, y2 = self._canvas_to_image_xy(event)
        self._template_drag_start = None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right <= left or bottom <= top:
            self.selected_template_roi = None
            self._set_status("Please select a valid template region.", "warning")
            return

        self.selected_template_roi = (left, top, right, bottom)
        if self._template_rect_id is not None:
            cx1, cy1 = self._image_to_canvas_xy(left, top)
            cx2, cy2 = self._image_to_canvas_xy(right, bottom)
            self.canvas.coords(self._template_rect_id, cx1, cy1, cx2, cy2)
        self.template_match_label.configure(
            text=f"Template selected:\nx1={left}, y1={top}\nx2={right}, y2={bottom}"
        )
        self._set_status("Template region selected. Click Save Selected Template.", "info")
    # Youssra-Phase2 END: template matching UI

    def _on_pan_start(self, event):
        """Begin click-drag panning."""
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        """Continue click-drag panning."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

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
    # Build sidebar sections
    # ─────────────────────────────────────────────────────────────────────────

    def build_sidebar_controls(self):
        # ── I/O ──────────────────────────────────────────────────────────────
        c = self._make_card("I / O  Controls")
        self.btn_load = self._op_btn(c, "Load Image", self.on_load_image, fg=CLR_SUCCESS, hover=CLR_SUCCESS_HVR)
        self.btn_save = self._op_btn(c, "Save Image", self.on_save_image, fg=CLR_CYAN,    hover=CLR_CYAN_HOVER)

        # ── Zoom ─────────────────────────────────────────────────────────────
        c = self._make_card("Zoom Controls")
        self._muted_label(c, "Scale Factor")
        self.scale_input = self._styled_entry(c, placeholder="e.g. 1.5")
        self._muted_label(c, "Interpolation Method")
        self.interp_type_var = ctk.StringVar(value="Nearest Neighbor")
        self.interp_dropdown = self._styled_optionmenu(
            c, ["Nearest Neighbor", "Linear"], self.interp_type_var,
        )
        self.btn_zoom = self._op_btn(c, "Apply Zoom", self.on_zoom)

        # Youssra-Phase2 START: geometric transform UI
        c = self._make_card("Geometric Transforms")
        self._muted_label(c, "Rotation Angle  (degrees)")
        self.rotation_angle_input = self._styled_entry(c, default="30")
        self.btn_rotate = self._op_btn(c, "Apply Rotation", self.on_apply_rotation)
        self._muted_label(c, "Shear X")
        self.shear_x_input = self._styled_entry(c, default="0.3")
        self._muted_label(c, "Shear Y")
        self.shear_y_input = self._styled_entry(c, default="0.0")
        self.btn_shear = self._op_btn(c, "Apply Shear", self.on_apply_shear)
        # Youssra-Phase2 END: geometric transform UI

        # ── Spatial Filters ───────────────────────────────────────────────────
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
        self.btn_sobel   = self._op_btn(c, "Sobel Edge",   lambda: self.on_edge_detection("sobel"))
        self.btn_prewitt = self._op_btn(c, "Prewitt Edge", lambda: self.on_edge_detection("prewitt"))

        # ── Contrast ──────────────────────────────────────────────────────────
        c = self._make_card("Contrast Enhancement")
        self._muted_label(c, "Block Size  (pixels)")
        self.block_size_input = self._styled_entry(c, placeholder="e.g. 32")
        self.btn_hist_eq = self._op_btn(c, "Apply Local Equalization", self.on_local_hist_eq)

        c = self._make_card("ROI / Noise Analysis")
        self._muted_label(c, "Noise Type")
        self.noise_type_var = ctk.StringVar(value="Gaussian")
        self.noise_dropdown = self._styled_optionmenu(c, ["Gaussian", "Uniform"], self.noise_type_var)
        self._muted_label(c, "Gaussian Mean")
        self.noise_gaussian_mean_input = self._styled_entry(c, default="0")
        self._muted_label(c, "Gaussian Std")
        self.noise_gaussian_std_input = self._styled_entry(c, default="10")
        self._muted_label(c, "Uniform Low")
        self.noise_uniform_low_input = self._styled_entry(c, default="-10")
        self._muted_label(c, "Uniform High")
        self.noise_uniform_high_input = self._styled_entry(c, default="10")
        self.btn_apply_noise = self._op_btn(
            c, "Apply Noise", self.on_apply_noise, fg=CLR_CYAN, hover=CLR_CYAN_HOVER,
        )
        self.btn_roi_stats = self._op_btn(c, "Calculate ROI Statistics", self.on_calculate_roi_statistics)
        self.btn_clear_roi = self._op_btn(
            c, "Clear ROI", self.clear_roi, fg=CLR_WARNING, hover=CLR_WARNING_HVR,
        )

        # Youssra-Phase2 START: template matching UI
        c = self._make_card("Template Matching")
        self.btn_template_mode = self._op_btn(
            c, "Select Template Mode", self.enter_template_selection_mode,
            fg=CLR_CYAN, hover=CLR_CYAN_HOVER,
        )
        self.btn_save_template = self._op_btn(c, "Save Selected Template", self.save_selected_template)
        self.btn_run_template = self._op_btn(c, "Run Template Matching", self.run_template_matching)
        self.btn_clear_template = self._op_btn(
            c, "Clear Template / Match Box", self.clear_template_match,
            fg=CLR_WARNING, hover=CLR_WARNING_HVR,
        )
        # Youssra-Phase2 END: template matching UI

        # ── Morphology ────────────────────────────────────────────────────────
        c = self._make_card("Morphology")

        # Thresholding (binarize before morphology operations)
        self._muted_label(c, "Threshold  (binarize image)")
        self.threshold_slider = ctk.CTkSlider(
            c, from_=0, to=255, number_of_steps=255,
            progress_color=CLR_ACCENT,
            button_color=CLR_TEXT_HDG,
            button_hover_color=CLR_TEXT_PRI,
        )
        self.threshold_slider.set(128)
        self.threshold_slider.pack(fill="x", pady=(2, 0))
        self.threshold_label = ctk.CTkLabel(
            c, text="Threshold: 128", font=FONT_LABEL, text_color=CLR_TEXT_SEC,
        )
        self.threshold_label.pack()
        self.threshold_slider.configure(
            command=lambda v: self.threshold_label.configure(text=f"Threshold: {int(v)}")
        )
        self.btn_threshold = self._op_btn(c, "Apply Threshold (Binarize)", self.on_threshold)

        # Structuring element
        self._muted_label(c, "Structuring Element  Shape")
        self.se_shape_var = ctk.StringVar(value="Square")
        self.se_dropdown  = self._styled_optionmenu(c, ["Square", "Cross"], self.se_shape_var)
        self._muted_label(c, "SE Size  (odd integer)")
        self.se_size_input = self._styled_entry(c, default="3")

        # Morphology operations
        self._muted_label(c, "Operations")
        self.btn_erode    = self._op_btn(c, "Erosion",             lambda: self.on_morphology("erode"))
        self.btn_dilate   = self._op_btn(c, "Dilation",            lambda: self.on_morphology("dilate"))
        self.btn_opening  = self._op_btn(c, "Opening",             lambda: self.on_morphology("open"))
        self.btn_closing  = self._op_btn(c, "Closing",             lambda: self.on_morphology("close"))
        self.btn_boundary = self._op_btn(c, "Boundary Extraction", lambda: self.on_morphology("boundary"))

        # ── Pipeline ──────────────────────────────────────────────────────────
        c = self._make_card("Pipeline Controls")
        self.btn_split = self._op_btn(c, "Toggle Split View", self.toggle_split_view, fg=CLR_CYAN, hover=CLR_CYAN_HOVER)

    # ─────────────────────────────────────────────────────────────────────────
    # Status / info helpers
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
            # undoable steps = history length minus the initial loaded entry
            undoable = max(0, len(self.image_history) - 1)
            self.dim_label.configure(text=f"{w} × {h}  {mode}  |  Undo: {undoable}  ")
            redoable = len(self.redo_history)
            self.history_label.configure(text=f"Undo: {undoable} | Redo: {redoable}")
        else:
            self.dim_label.configure(text="")
            self.history_label.configure(text="Undo: 0 | Redo: 0")

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
    # Input validation
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

    def _parse_float(self, widget, name):
        try:
            return float(widget.get().strip())
        except ValueError:
            self._set_status(f"{name}: enter a valid number.", "error")
            return None

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
            self._set_status("Please load an image first.", "warning")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Core display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def show_error(self, message):
        self._set_status(message[:90], "error")
        messagebox.showerror("Error", message)

    def show_info(self, message):
        self._set_status(message[:90], "success")
        messagebox.showinfo("Success", message)

    def _bind_keyboard_shortcuts(self):
        self.master.bind_all("<Control-z>", lambda event: self.on_undo())
        self.master.bind_all("<Control-y>", lambda event: self.on_redo())
        self.master.bind_all("<Control-r>", lambda event: self.on_reset())

    # Owner: Zeyad - GUI / Pipeline Architecture
    def _set_ai_status(self, message, level="info"):
        colors = {
            "info": CLR_TEXT_SEC,
            "success": CLR_AI_ACCENT,
            "warning": "#d29922",
            "error": "#f85149",
            "busy": CLR_TEXT_HDG,
        }
        self.ai_status_label.configure(text=message, text_color=colors.get(level, CLR_TEXT_SEC))
        self._set_status(f"AI: {message}", level if level in {"success", "warning", "error", "busy"} else "info")
        self.master.update_idletasks()

    def _set_ai_diagnosis(self, message):
        self.ai_diagnosis_text.configure(state="normal")
        self.ai_diagnosis_text.delete("1.0", "end")
        self.ai_diagnosis_text.insert("end", message)
        self.ai_diagnosis_text.configure(state="disabled")

    # AI-Segmentation START
    def _update_ai_info(self):
        confidence = "--" if self.ai_confidence is None else f"{self.ai_confidence:.3f}"
        self.ai_info_label.configure(
            text=(
                f"Model loaded: {'Yes' if self.ai_model is not None else 'No'}\n"
                f"AI image ready: {'Yes' if self.ai_image is not None else 'No'}\n"
                f"AI model: {'Ready' if self.ai_model is not None else 'Not loaded'}\n"
                f"AI image mode: {self.ai_image_mode or '--'}\n"
                f"AI input source: {self.ai_input_source or '--'}\n"
                f"Mask ready: {'Yes' if self.ai_mask is not None else 'No'}\n"
                f"Confidence: {confidence}"
            )
        )
    # AI-Segmentation END

    # AI-Segmentation START: UX simplification
    def _set_ai_completion_diagnosis(self, result):
        self._set_ai_diagnosis(
            "AI segmentation completed.\n"
            f"Confidence: {self.ai_confidence:.3f}\n"
            f"Threshold: {result['threshold']}\n"
            f"Model input: {result['img_size']} x {result['img_size']}\n\n"
            "Meaning:\n"
            "Confidence is the average predicted probability inside the mask.\n"
            f"Threshold means pixels above {result['threshold']} become foreground.\n"
            "The AI result is displayed temporarily. Pipeline history is unchanged.\n\n"
            "This model is trained for endoscopy polyp segmentation. Results on "
            "non-endoscopy images are only a technical test and are not clinically meaningful."
        )

    def _ensure_ai_original_preview(self):
        self._set_ai_preview_image("original", self.ai_image, "No AI image loaded")
    # AI-Segmentation END: UX simplification

    # AI-Segmentation START: PhotoImage reference fix
    # AI-Segmentation START: persistent PhotoImage fix
    # AI-Segmentation START: complete image reference audit
    def _get_ai_preview_widget(self, target):
        if target == "original":
            return self.ai_original_preview, (520, 360)
        if target == "overlay":
            return self.ai_overlay_preview, (520, 360)
        if target == "mask":
            return self.ai_mask_preview, (420, 170)
        raise ValueError(f"Unknown AI preview target: {target}")

    def _set_ai_preview_reference(self, target, photo):
        if target == "original":
            self.ai_original_photo = photo
            self.ai_original_ctk_image = None
            self.ai_original_image_id = None
            self.ai_original_canvas_item = None
            return self.ai_original_photo
        if target == "overlay":
            self.ai_overlay_photo = photo
            self.ai_overlay_ctk_image = None
            self.ai_overlay_image_id = None
            self.ai_overlay_canvas_item = None
            return self.ai_overlay_photo
        if target == "mask":
            self.ai_mask_photo = photo
            self.ai_mask_ctk_image = None
            self.ai_mask_image_id = None
            self.ai_mask_canvas_item = None
            return self.ai_mask_photo
        return photo

    def _clear_ai_preview_reference(self, target):
        if target == "original":
            self.ai_original_photo = None
            self.ai_original_ctk_image = None
            self.ai_original_image_id = None
            self.ai_original_canvas_item = None
        elif target == "overlay":
            self.ai_overlay_photo = None
            self.ai_overlay_ctk_image = None
            self.ai_overlay_image_id = None
            self.ai_overlay_canvas_item = None
        elif target == "mask":
            self.ai_mask_photo = None
            self.ai_mask_ctk_image = None
            self.ai_mask_image_id = None
            self.ai_mask_canvas_item = None

    def _normalise_ai_preview_array(self, image_array):
        arr = np.asarray(image_array)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return np.clip(arr, 0, 255).astype(np.uint8)

    # AI-Segmentation START: robust upload/analyze/clear fix
    def _set_ai_preview(self, target, image_array_or_none, placeholder_text=None):
        """Single safe entry-point for every AI preview update.

        Safety contract (canvas-based implementation):
          1. canvas.delete("all") is called FIRST — this synchronously removes the
             Tk-level reference to any previously displayed PhotoImage, before any
             Python-side reference is released.  The old PhotoImage can now be GC'd
             without Tkinter ever seeing a dangling image name.
          2. New PhotoImage is stored on self.* AND on canvas._ai_photo before
             canvas.create_image() is called, so the photo is always alive while Tk
             holds it.
          This eliminates the 'pyimageX doesn't exist' error class entirely.
        """
        canvas, fallback_size = self._get_ai_preview_widget(target)
        if placeholder_text is None:
            placeholder_text = {"original": "No AI image loaded",
                                 "overlay": "Overlay preview",
                                 "mask": "Mask preview"}.get(target, "")

        # ── Step 1: unconditionally wipe the canvas (removes Tk image ref) ──────
        canvas.delete("all")
        canvas._ai_photo = None

        # ── Step 2: clear Python-side references for this slot ──────────────────
        self._clear_ai_preview_reference(target)
        self._ai_preview_photos.pop(target, None)

        if image_array_or_none is None:
            # Show placeholder text centered in whatever space is available.
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w > 1 and h > 1:
                canvas.create_text(
                    w // 2, h // 2,
                    text=placeholder_text,
                    fill=CLR_TEXT_SEC,
                    font=FONT_LABEL,
                    justify="center",
                )
            canvas._ai_placeholder = placeholder_text
            return

        # ── Step 3: build the new PhotoImage ────────────────────────────────────
        arr = self._normalise_ai_preview_array(image_array_or_none)
        pil_img = Image.fromarray(arr)

        # Fit to canvas without upscaling.
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        max_w = cw if cw > 1 else fallback_size[0]
        max_h = ch if ch > 1 else fallback_size[1]
        iw, ih = pil_img.size
        scale = min(max_w / max(1, iw), max_h / max(1, ih), 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize(
                (max(1, int(iw * scale)), max(1, int(ih * scale))),
                Image.LANCZOS,
            )

        new_photo = ImageTk.PhotoImage(image=pil_img)

        # ── Step 4: store BEFORE any canvas operation ────────────────────────────
        # All three anchors must be in place so CPython's reference counter never
        # drops the photo to zero while the canvas item holds the Tk image name.
        self._set_ai_preview_reference(target, new_photo)   # self.ai_*_photo = new_photo
        self.ai_display_photo = new_photo
        self._ai_preview_photos[target] = new_photo
        canvas._ai_photo = new_photo   # anchor on the canvas widget itself

        # ── Step 5: draw ─────────────────────────────────────────────────────────
        cx = max_w // 2
        cy = max_h // 2
        canvas.create_image(cx, cy, image=new_photo, anchor="center")
        canvas._ai_placeholder = None

    def _set_ai_preview_image(self, target, image_array_or_none, placeholder_text=None):
        """Backward-compatible alias — routes to _set_ai_preview."""
        self._set_ai_preview(target, image_array_or_none, placeholder_text)
    # AI-Segmentation END: robust upload/analyze/clear fix

    def _ai_preview_slot(self, label):
        if label is getattr(self, "ai_original_preview", None):
            return "original"
        if label is getattr(self, "ai_overlay_preview", None):
            return "overlay"
        if label is getattr(self, "ai_mask_preview", None):
            return "mask"
        return None

    def _display_ai_preview(self, label, image_array, key=None, max_size=None):
        target = self._ai_preview_slot(label)
        if target is None:
            return
        placeholders = {
            "original": "No AI image loaded",
            "overlay": "Overlay preview",
            "mask": "Mask preview",
        }
        self._set_ai_preview_image(target, image_array, placeholders[target])
    # AI-Segmentation END: persistent PhotoImage fix
    # AI-Segmentation END: complete image reference audit
    # AI-Segmentation END: PhotoImage reference fix

    def _parse_ai_video_limits(self):
        try:
            frame_step = max(1, int(self.ai_frame_step_input.get().strip()))
            max_frames = max(1, int(self.ai_max_frames_input.get().strip()))
        except ValueError:
            self._set_ai_status("Video limits must be positive integers.", "error")
            return None, None
        return frame_step, max_frames

    def _require_ai_model(self):
        if self.ai_model is None:
            self._set_ai_status("Load an AI model first.", "warning")
            return False
        return True

    def refresh_canvas(self, new_numpy_array):
        """Render a NumPy array on the tk.Canvas at 1:1 pixel ratio.

        The array is never downscaled — scrollbars let the user pan around
        images larger than the viewport.
        
        Garbage Collection Prevention:
        Tkinter's C-backend does not hold a reference to Python-created images. 
        If the `ImageTk.PhotoImage` is assigned to a local variable, Python's 
        garbage collector will destroy it when this function exits, resulting 
        in a blank/white canvas. We prevent this by anchoring the image 
        reference to `self._photo_image`, forcing it to stay alive in memory 
        as long as the class instance exists.
        """
        if new_numpy_array is None:
            return

        if getattr(self, "is_split_view", False):
            self._update_split_view()

        # Normalize to uint8 for PIL
        if new_numpy_array.dtype != np.uint8:
            mn, mx = new_numpy_array.min(), new_numpy_array.max()
            if mx > mn:
                norm = ((new_numpy_array - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(new_numpy_array, dtype=np.uint8)
        else:
            norm = new_numpy_array

        pil_img = Image.fromarray(norm)

        # Create PhotoImage (must be kept alive in self._photo_image)
        new_photo = ImageTk.PhotoImage(image=pil_img)
        self._photo_image = new_photo   # assign before canvas uses it

        # Remove placeholder text
        if self._canvas_placeholder_id is not None:
            self.canvas.delete(self._canvas_placeholder_id)
            self._canvas_placeholder_id = None

        # Update or create canvas image item
        if self._canvas_image_id is not None:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo_image)
        else:
            self._canvas_image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._photo_image,
            )

        # Expand scroll region to the full image size so scrollbars reflect reality
        img_w, img_h = pil_img.size
        arr_h, arr_w = new_numpy_array.shape[:2]
        self._display_image_origin = (0, 0)
        self._display_image_size = (img_w, img_h)
        self._display_to_array_scale = (
            arr_w / img_w if img_w else 1.0,
            arr_h / img_h if img_h else 1.0,
        )
        self.canvas.configure(scrollregion=(0, 0, img_w, img_h))
        if self._roi_rect_id is not None:
            self.canvas.tag_raise(self._roi_rect_id)
        if self._template_rect_id is not None:
            self.canvas.tag_raise(self._template_rect_id)
        if self._match_box_id is not None:
            self.canvas.tag_raise(self._match_box_id)

        # Reset viewport to top-left on every new result
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

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

    # ─────────────────────────────────────────────────────────────────────────
    # State management core
    # ─────────────────────────────────────────────────────────────────────────

    def _commit_result(self, result):
        """Store result as new current state, append to history, refresh display.

        This is the single place where image_history is written to after load,
        ensuring history never contains None and always has valid copies.
        """
        if result is None or not isinstance(result, np.ndarray) or result.size == 0:
            return False

        safe_copy = result.copy()
        self.current_image_array = safe_copy
        self.image_history.append(safe_copy.copy())
        self.redo_history.clear()

        h, w = safe_copy.shape[:2]
        self.metadata_dict["Width"]  = str(w)
        self.metadata_dict["Height"] = str(h)
        self.update_metadata_panel(self.metadata_dict)
        self.refresh_canvas(safe_copy)
        # Youssra-Phase2: processing operations change the image, so stale template overlays are cleared.
        self.clear_template_match(show_message=False, clear_template=True)
        return True

    def apply_action(self, action_func, *args, **kwargs):
        """Run action_func on current_image_array.  On success commit result.
        Returns True on success, False on failure (state is unchanged on failure).
        """
        if self.current_image_array is None:
            self._set_status("Please load an image first.", "warning")
            return False
        try:
            new_image = action_func(self.current_image_array, *args, **kwargs)
            if new_image is None:
                self._set_status("This operation is not yet implemented.", "warning")
                return False
            return self._commit_result(new_image)
        except Exception as e:
            self._set_status(f"Error: {str(e)[:70]}", "error")
            messagebox.showerror("Engine Error", str(e))
            return False

    def _apply_compound_morph(self, funcs, se, label):
        """Apply a SEQUENCE of morphology functions as ONE pipeline step.

        Used for Opening (erode → dilate) and Closing (dilate → erode).
        The entire compound operation counts as a single undo step.
        """
        if self.current_image_array is None:
            return False
        try:
            img = self.current_image_array
            for fn in funcs:
                intermediate = fn(img, se)
                if intermediate is None:
                    self._set_status(f"{label}: a sub-step is not yet implemented.", "warning")
                    return False
                img = intermediate
            return self._commit_result(img)
        except Exception as e:
            self._set_status(f"Error in {label}: {str(e)[:60]}", "error")
            messagebox.showerror("Engine Error", str(e))
            return False

    def _apply_boundary_extraction(self, se, label):
        """Boundary = current_image  −  erode(current_image, se).

        Implemented here in the UI layer so no engine TODO is needed.
        """
        if self.current_image_array is None:
            return False
        try:
            img = self.current_image_array
            eroded = morphology_engine.erode(img, se)
            if eroded is None:
                self._set_status("Boundary Extraction: erosion step not available.", "warning")
                return False

            if img.dtype == np.uint8 and eroded.dtype == np.uint8:
                boundary = np.clip(
                    img.astype(np.int16) - eroded.astype(np.int16), 0, 255
                ).astype(np.uint8)
            else:
                # Boolean arrays — XOR keeps only the boundary ring
                boundary = (img.astype(bool) & ~eroded.astype(bool)).astype(np.uint8) * 255

            return self._commit_result(boundary)
        except Exception as e:
            self._set_status(f"Error in {label}: {str(e)[:60]}", "error")
            messagebox.showerror("Engine Error", str(e))
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks  (names unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    # Bahr-Phase2 START: ROI histogram display
    def _draw_roi_histogram(self, histogram):
        """Draw a compact 64-bar view of the 256-bin ROI histogram."""
        if not hasattr(self, "roi_hist_canvas"):
            return

        canvas = self.roi_hist_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width() or 220)
        height = max(1, canvas.winfo_height() or 96)
        pad_x = 8
        pad_y = 8

        canvas.create_line(
            pad_x, height - pad_y, width - pad_x, height - pad_y,
            fill=CLR_BORDER,
        )
        canvas.create_text(
            pad_x,
            pad_y,
            text="ROI Histogram",
            fill=CLR_TEXT_SEC,
            font=FONT_MONO,
            anchor="nw",
        )

        if histogram is None:
            canvas.create_text(
                width // 2,
                height // 2 + 8,
                text="No ROI data",
                fill=CLR_TEXT_SEC,
                font=FONT_MONO,
            )
            return

        hist = np.asarray(histogram, dtype=np.float64)
        if hist.size != 256 or np.max(hist) <= 0:
            return

        grouped = hist.reshape(64, 4).sum(axis=1)
        max_count = np.max(grouped)
        if max_count <= 0:
            return

        chart_top = 24
        chart_bottom = height - pad_y - 1
        chart_h = max(1, chart_bottom - chart_top)
        chart_w = max(1, width - 2 * pad_x)
        bar_w = chart_w / 64.0

        for idx, count in enumerate(grouped):
            x1 = pad_x + idx * bar_w
            x2 = pad_x + (idx + 1) * bar_w - 1
            bar_h = (count / max_count) * chart_h
            y1 = chart_bottom - bar_h
            canvas.create_rectangle(
                x1, y1, max(x1 + 1, x2), chart_bottom,
                fill=CLR_CYAN,
                outline="",
            )
    # Bahr-Phase2 END: ROI histogram display

    # Youssra-Phase2 START: geometric transform UI
    def on_apply_rotation(self):
        if not self._require_image():
            return
        angle = self._parse_float(self.rotation_angle_input, "Rotation angle")
        if angle is None:
            return

        self._set_status(f"Applying rotation ({angle:.2f} degrees) ...", "busy")
        ok = self.apply_action(spatial_engine.rotate_image, angle)
        if ok:
            self.clear_template_match(show_message=False, clear_template=True)
            self._set_status(f"Rotation applied ({angle:.2f} degrees).", "success")

    def on_apply_shear(self):
        if not self._require_image():
            return
        shear_x = self._parse_float(self.shear_x_input, "Shear X")
        shear_y = self._parse_float(self.shear_y_input, "Shear Y")
        if shear_x is None or shear_y is None:
            return

        self._set_status(f"Applying shear (x={shear_x:.2f}, y={shear_y:.2f}) ...", "busy")
        ok = self.apply_action(spatial_engine.shear_image, shear_x, shear_y)
        if ok:
            self.clear_template_match(show_message=False, clear_template=True)
            self._set_status(f"Shear applied (x={shear_x:.2f}, y={shear_y:.2f}).", "success")
    # Youssra-Phase2 END: geometric transform UI

    # Youssra-Phase2 START: template matching UI
    def enter_template_selection_mode(self):
        if not self._require_image():
            return
        self.selection_mode = "template"
        self.template_match_label.configure(text="Mode: template selection\ndrag on image")
        self._set_status("Template selection mode: drag a rectangle on the image.", "info")

    def save_selected_template(self):
        if not self._require_image():
            return
        if self.selected_template_roi is None:
            self._set_status("Please select a template region first.", "warning")
            return

        x1, y1, x2, y2 = self.selected_template_roi
        if x2 <= x1 or y2 <= y1:
            self._set_status("Please select a valid template region.", "warning")
            return

        template = self.current_image_array[y1:y2, x1:x2]
        if template.size == 0:
            self._set_status("Please select a valid template region.", "warning")
            return

        self.selected_template = template.copy()
        self.selection_mode = "roi"
        h, w = self.selected_template.shape[:2]
        self.template_match_label.configure(
            text=f"Template saved:\nSize: {w} x {h}\nMode: ROI statistics"
        )
        self._set_status("Template saved. ROI statistics mode restored.", "success")

    def _draw_template_match_box(self, result):
        if self._match_box_id is not None:
            self.canvas.delete(self._match_box_id)
            self._match_box_id = None

        x1, y1 = result["top_left"]
        x2, y2 = result["bottom_right"]
        cx1, cy1 = self._image_to_canvas_xy(x1, y1)
        cx2, cy2 = self._image_to_canvas_xy(x2, y2)
        self._match_box_id = self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=CLR_DANGER,
            width=3,
        )
        self.canvas.tag_raise(self._match_box_id)

    def run_template_matching(self):
        if not self._require_image():
            return
        if self.selected_template is None:
            self._set_status("Please select a template first.", "warning")
            return

        self._set_status("Running Fourier template matching ...", "busy")
        try:
            result = frequency_engine.template_matching_fourier(
                self.current_image_array,
                self.selected_template,
            )
        except ValueError as e:
            self._set_status(str(e), "warning")
            return
        except Exception as e:
            self._set_status(f"Template matching error: {str(e)[:60]}", "error")
            return

        self.last_template_match = result
        self._draw_template_match_box(result)
        x1, y1 = result["top_left"]
        x2, y2 = result["bottom_right"]
        self.template_match_label.configure(
            text=(
                f"Match box:\n"
                f"({x1}, {y1}) to ({x2}, {y2})\n"
                f"Score: {result['score']:.2f}"
            )
        )
        self._set_status("Template match complete. Bounding box displayed.", "success")

    def clear_template_match(self, show_message=True, clear_template=True):
        self.selected_template_roi = None
        self.last_template_match = None
        self._template_drag_start = None
        if clear_template:
            self.selected_template = None
        if self._template_rect_id is not None:
            self.canvas.delete(self._template_rect_id)
            self._template_rect_id = None
        if self._match_box_id is not None:
            self.canvas.delete(self._match_box_id)
            self._match_box_id = None
        self.selection_mode = "roi"
        if hasattr(self, "template_match_label"):
            self.template_match_label.configure(text="Mode: ROI statistics")
        if show_message:
            self._set_status("Template selection and match box cleared.", "info")
    # Youssra-Phase2 END: template matching UI

    # Bahr-Phase2
    def clear_roi(self, show_message=True):
        """Clear the selected ROI rectangle and displayed ROI statistics."""
        self.selected_roi = None
        self.last_roi_histogram = None
        self._roi_drag_start = None
        if self._roi_rect_id is not None:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
        self.roi_stats_label.configure(text="Draw a rectangle on the image.")
        self._draw_roi_histogram(None)
        if show_message:
            self._set_status("ROI cleared.", "info")

    # Bahr-Phase2
    def on_apply_noise(self):
        if not self._require_image():
            return

        noise_type = self.noise_type_var.get()
        if noise_type == "Gaussian":
            mean = self._parse_float(self.noise_gaussian_mean_input, "Gaussian mean")
            std = self._parse_float(self.noise_gaussian_std_input, "Gaussian std")
            if mean is None or std is None:
                return
            parameters = {"mean": mean, "std": std}
        else:
            low = self._parse_float(self.noise_uniform_low_input, "Uniform low")
            high = self._parse_float(self.noise_uniform_high_input, "Uniform high")
            if low is None or high is None:
                return
            parameters = {"low": low, "high": high}

        self._set_status(f"Applying {noise_type} noise ...", "busy")
        ok = self.apply_action(frequency_engine.inject_noise, noise_type, parameters)
        if ok:
            self._set_status(f"{noise_type} noise applied.", "success")

    # Bahr-Phase2
    def on_calculate_roi_statistics(self):
        if not self._require_image():
            return
        if self.selected_roi is None:
            self._set_status("Please select a valid ROI.", "warning")
            return

        try:
            stats = frequency_engine.calculate_roi_statistics(
                self.current_image_array,
                self.selected_roi,
            )
        except ValueError as e:
            self._set_status(str(e), "warning")
            return
        except Exception as e:
            self._set_status(f"ROI statistics error: {str(e)[:60]}", "error")
            return

        self.last_roi_histogram = stats["histogram"]
        self._draw_roi_histogram(self.last_roi_histogram)
        self.roi_stats_label.configure(
            text=(
                f"Mean:   {stats['mean']:.2f}\n"
                f"Var:    {stats['variance']:.2f}\n"
                f"Std:    {stats['std']:.2f}\n"
                f"Min:    {stats['min']}\n"
                f"Max:    {stats['max']}\n"
                f"Pixels: {stats['pixel_count']}"
            )
        )
        print(
            "ROI Statistics | "
            f"mean={stats['mean']:.2f}, variance={stats['variance']:.2f}, "
            f"std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}, "
            f"pixels={stats['pixel_count']}"
        )
        self._set_status("ROI statistics calculated and histogram displayed.", "success")

    # AI-Segmentation START
    # AI-Segmentation START: cleanup and clear bug fix
    # AI-Segmentation START: PhotoImage reference fix
    # AI-Segmentation START: persistent PhotoImage fix
    # AI-Segmentation START: complete image reference audit
    def _clear_ai_preview_label(self, label, text, clear_refs=False):
        target = self._ai_preview_slot(label)
        if target is not None:
            self._set_ai_preview(target, None, text)
        else:
            # Unknown widget — try CTkLabel path, then canvas fallback.
            try:
                label.configure(image=None, text=text)
                label.image = None
            except Exception:
                try:
                    label.delete("all")
                except Exception:
                    pass
        if clear_refs:
            self._ai_preview_photos.clear()
            self.ai_original_photo = None
            self.ai_overlay_photo = None
            self.ai_mask_photo = None
            self.ai_display_photo = None
            self.ai_original_ctk_image = None
            self.ai_overlay_ctk_image = None
            self.ai_mask_ctk_image = None
            self.ai_original_image_id = None
            self.ai_overlay_image_id = None
            self.ai_mask_image_id = None
            self.ai_original_canvas_item = None
            self.ai_overlay_canvas_item = None
            self.ai_mask_canvas_item = None
    # AI-Segmentation END: complete image reference audit
    # AI-Segmentation END: persistent PhotoImage fix
    # AI-Segmentation END: PhotoImage reference fix
    # AI-Segmentation END: cleanup and clear bug fix

    # AI-Segmentation START: auto-load and RGB fix
    def _ensure_ai_model_loaded(self):
        if self.ai_model is not None:
            return True
        self._set_ai_status("Loading AI model...", "busy")
        try:
            self.ai_model = ai_segmentation_engine.load_model(model_path=self.ai_model_path)
            config = ai_segmentation_engine.load_ai_config()
            self.ai_model_path = config.get("model_path", self.ai_model_path)
            self._update_ai_info()
            self._set_ai_status("AI model loaded.", "success")
            return True
        except Exception as e:
            self.ai_model = None
            self._update_ai_info()
            self._set_ai_status("AI model error.", "error")
            messagebox.showerror("AI Model Load", str(e))
            return False

    # AI-Segmentation START: robust upload/analyze/clear fix
    def on_ai_upload_image(self):
        path = filedialog.askopenfilename(
            title="Select RGB Image for AI",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            # Always use Pillow direct load + convert("RGB") — never image_io
            # which may apply grayscale conversion or alter the array.
            pil_img = Image.open(path).convert("RGB")
            rgb_array = np.array(pil_img, dtype=np.uint8).copy()  # owned copy

            # ── Reset all previous AI outputs BEFORE assigning new image ─────────
            self.ai_mask = None
            self.ai_overlay = None
            self.ai_confidence = None
            self.ai_display_mode = None
            self.ai_image_was_grayscale = False

            # ── Assign new image ─────────────────────────────────────────────────
            self.ai_image = rgb_array
            self.ai_rgb_file_path = path
            self.ai_input_source = "AI-only RGB upload"
            self.ai_image_mode = "RGB"

            # ── Refresh all three previews through the single safe helper ────────
            self._set_ai_preview("original", self.ai_image)
            self._set_ai_preview("overlay",  None)
            self._set_ai_preview("mask",     None)

            h, w = self.ai_image.shape[:2]
            self._update_ai_info()
            self._set_ai_diagnosis(
                "RGB image loaded directly for AI inference.\n"
                f"Size: {w} x {h} px\n\n"
                "Normal pipeline image/history unchanged."
            )
            self._set_ai_status("RGB image loaded for AI.", "success")
        except Exception as e:
            self._set_ai_status("AI RGB image load failed.", "error")
            messagebox.showerror("AI Image Load", str(e))
    # AI-Segmentation END: robust upload/analyze/clear fix
    # AI-Segmentation END: auto-load and RGB fix

    def on_ai_load_model(self):
        self._set_ai_status("Loading Keras AI model...", "busy")
        try:
            self.ai_model = ai_segmentation_engine.load_model(model_path=self.ai_model_path)
            config = ai_segmentation_engine.load_ai_config()
            self.ai_model_path = config.get("model_path", self.ai_model_path)
            if hasattr(self, "ai_threshold_var"):
                self.ai_threshold_var.set(float(config.get("threshold", self.ai_threshold_var.get())))
                self.ai_threshold_label.configure(text=f"Threshold: {self.ai_threshold_var.get():.2f}")
            self._update_ai_info()
            self._set_ai_diagnosis(
                "AI model loaded\n"
                f"Model: {self.ai_model_path}\n"
                f"Input size: {config['img_size']} x {config['img_size']}\n"
                f"Threshold: {self.ai_threshold_var.get() if hasattr(self, 'ai_threshold_var') else config['threshold']}\n\n"
                "This model is trained for endoscopy polyp segmentation. Results on non-endoscopy images are only a technical test and are not clinically meaningful."
            )
            self._set_ai_status("AI model loaded.", "success")
        except Exception as e:
            self.ai_model = None
            self._update_ai_info()
            self._set_ai_status("AI model load failed.", "error")
            messagebox.showerror("AI Model Load", str(e))

    def on_ai_use_current_image(self):
        if not self._require_image():
            return
        # AI-Segmentation START: auto-load and RGB fix
        source_note = "Pipeline"
        mode_note = "RGB"
        if self.current_image_array.ndim == 3 and self.current_image_array.shape[2] >= 3:
            self.ai_image = self.current_image_array[:, :, :3].copy()
        elif (
            self.original_image_array is not None
            and self.original_image_array.ndim == 3
            and self.original_image_array.shape[2] >= 3
        ):
            self.ai_image = self.original_image_array[:, :, :3].copy()
            source_note = "Original RGB"
        else:
            self.ai_image = self.current_image_array.copy()
            mode_note = "Grayscale converted to RGB"
        self.ai_input_source = source_note
        self.ai_image_mode = mode_note
        # AI-Segmentation END: auto-load and RGB fix
        self.ai_mask = None
        self.ai_overlay = None
        self.ai_confidence = None
        self.ai_display_mode = None
        self.ai_video_results = []
        # AI-Segmentation START: analyze rendering fix
        self._set_ai_preview("original", self.ai_image, "No AI image loaded")
        self._set_ai_preview("mask", None, "Mask preview")
        self._set_ai_preview("overlay", None, "Overlay preview")
        # AI-Segmentation END: analyze rendering fix
        h, w = self.ai_image.shape[:2]
        self._update_ai_info()
        self._set_ai_diagnosis(
            "Current image copied to AI workspace\n"
            f"Size: {w} x {h} px\n\n"
            "Pipeline image/history unchanged.\n"
            f"AI input source: {self.ai_input_source}\n"
            f"AI image mode: {self.ai_image_mode}\n\n"
            + ("Pipeline grayscale image converted to RGB for model compatibility.\n\n" if self.ai_image_mode != "RGB" else "")
            +
            "This model is trained for endoscopy polyp segmentation. Results on non-endoscopy images are only a technical test and are not clinically meaningful."
        )
        self._set_ai_status("Current image copied to AI workspace.", "success")
        self.main_tabs.set("Polyp AI Detection")

    # AI-Segmentation START: UX simplification
    def on_ai_run_current_image(self):
        if not self._ensure_ai_model_loaded():
            return
        if self.ai_image is None:
            if not self._require_image():
                return
            self.on_ai_use_current_image()
        self.on_ai_analyze_image()
    # AI-Segmentation END: UX simplification

    def on_ai_analyze_image(self):
        if self.ai_image is None:
            self._set_ai_status("Please upload an AI image first.", "warning")
            return
        if not self._ensure_ai_model_loaded():
            return
        self._set_ai_status("Running AI segmentation...", "busy")
        try:
            result = ai_segmentation_engine.run_inference(
                self.ai_image,
                model_path=self.ai_model_path,
                threshold=self.ai_threshold_var.get() if hasattr(self, "ai_threshold_var") else None,
            )
            self.ai_model = ai_segmentation_engine.load_model(model_path=self.ai_model_path)
            self.ai_mask = result["mask"]
            self.ai_overlay = result["overlay"]
            self.ai_confidence = result["confidence"]
            self.ai_display_mode = "overlay"
            # AI-Segmentation START: analyze rendering fix
            # All three previews go through the single safe helper.
            # "original" is set first so it is never omitted even when mask/overlay fail.
            self._set_ai_preview("original", self.ai_image, "No AI image loaded")
            self._set_ai_preview("mask", self.ai_mask, "Mask preview")
            self._set_ai_preview("overlay", self.ai_overlay, "Overlay preview")
            # AI-Segmentation END: analyze rendering fix
            self._update_ai_info()
            self._set_ai_completion_diagnosis(result)
            self._set_ai_status("AI segmentation completed.", "success")
        except Exception as e:
            self._set_ai_status("AI segmentation failed.", "error")
            messagebox.showerror("AI Segmentation", str(e))

    def on_ai_show_mask(self):
        if self.ai_mask is None:
            self._set_ai_status("No AI mask available yet.", "warning")
            return
        self.ai_display_mode = "mask"
        # AI-Segmentation START: analyze rendering fix
        self._set_ai_preview("original", self.ai_image, "No AI image loaded")
        self._set_ai_preview("mask", self.ai_mask, "Mask preview")
        self._set_ai_preview("overlay", self.ai_mask, "Overlay preview")
        # AI-Segmentation END: analyze rendering fix
        self._set_ai_status("AI mask displayed.", "info")

    def on_ai_show_overlay(self):
        if self.ai_overlay is None:
            self._set_ai_status("No AI overlay available yet.", "warning")
            return
        self.ai_display_mode = "overlay"
        # AI-Segmentation START: analyze rendering fix
        self._set_ai_preview("original", self.ai_image, "No AI image loaded")
        self._set_ai_preview("overlay", self.ai_overlay, "Overlay preview")
        # AI-Segmentation END: analyze rendering fix
        self._set_ai_status("AI overlay displayed.", "info")

    # AI-Segmentation START: UX simplification
    def on_ai_toggle_view(self):
        if self.ai_mask is None or self.ai_overlay is None:
            self._set_ai_status("Run AI segmentation first.", "warning")
            return
        if self.ai_display_mode == "mask":
            self.on_ai_show_overlay()
        else:
            self.on_ai_show_mask()
    # AI-Segmentation END: UX simplification

    def on_ai_back_to_pipeline(self):
        if not self._require_image():
            return
        self.ai_display_mode = None
        self.refresh_canvas(self.current_image_array)
        self._set_ai_status("Back to pipeline image.", "info")

    def on_ai_save_result(self):
        if self.ai_mask is None or self.ai_overlay is None:
            self._set_ai_status("No AI result available to save.", "warning")
            return

        save_mask = self.ai_display_mode == "mask"
        default_name = "polyp_segmentation_mask.png" if save_mask else "polyp_segmentation_overlay.png"
        path = filedialog.asksaveasfilename(
            title="Save AI Result",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("All Files", "*.*")],
        )
        if not path:
            return

        try:
            array = self.ai_mask if save_mask else self.ai_overlay
            Image.fromarray(np.asarray(array).astype(np.uint8)).save(path)
            self._set_ai_status("AI result saved.", "success")
            messagebox.showinfo("AI Result Saved", f"Saved AI result to:\n{path}")
        except Exception as e:
            self._set_ai_status("Save failed.", "error")
            messagebox.showerror("AI Save Result", str(e))

    # AI-Segmentation START: robust upload/analyze/clear fix
    def on_ai_clear_workspace(self):
        """Reset all AI state and visuals without touching the pipeline."""
        # ── State reset ──────────────────────────────────────────────────────────
        self.ai_image = None
        self.ai_mask = None
        self.ai_overlay = None
        self.ai_confidence = None
        self.ai_display_mode = None
        self.ai_input_source = None
        self.ai_image_mode = None
        self.ai_rgb_file_path = None
        self.ai_image_was_grayscale = False
        self.ai_video_results = []

        # ── Visual reset — _set_ai_preview calls canvas.delete("all") first, ────
        # which removes the Tk image reference before Python refs are released.
        self._set_ai_preview("original", None)
        self._set_ai_preview("overlay",  None)
        self._set_ai_preview("mask",     None)

        # ── Drop all remaining Python photo references ────────────────────────
        # (_set_ai_preview already called _clear_ai_preview_reference for each
        # target, but we also wipe the shared dict and display anchor.)
        self._ai_preview_photos.clear()
        self.ai_display_photo   = None
        self.ai_original_photo  = None
        self.ai_overlay_photo   = None
        self.ai_mask_photo      = None

        self._update_ai_info()
        self._set_ai_diagnosis(
            "AI workspace cleared.\n"
            "Pipeline image/history unchanged.\n\n"
            "This model is trained for endoscopy polyp segmentation. "
            "Results on non-endoscopy images are only a technical test "
            "and are not clinically meaningful."
        )
        self._set_ai_status("AI workspace cleared.", "info")
    # AI-Segmentation END: robust upload/analyze/clear fix

    def on_ai_apply_overlay_to_pipeline(self):
        if self.ai_overlay is None:
            self._set_ai_status("No AI overlay available to apply.", "warning")
            return
        if self._commit_result(self.ai_overlay.copy()):
            self.ai_display_mode = None
            self._set_ai_status("AI overlay applied to pipeline.", "success")
    # AI-Segmentation END

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

            # Initialise state — history starts with the loaded image
            self.original_image_array = image_array.copy()
            self.current_image_array  = image_array.copy()
            self.image_history        = [image_array.copy()]   # ← spec: [loaded_copy]
            self.redo_history         = []
            self.metadata_dict        = metadata
            self.clear_roi(show_message=False)
            # Youssra-Phase2: a new image invalidates any saved template or match box.
            self.clear_template_match(show_message=False, clear_template=True)

            self.refresh_canvas(self.current_image_array)
            self.update_metadata_panel(self.metadata_dict)
            self._set_status("Image loaded successfully.", "success")
        except Exception as e:
            self.show_error(f"Error loading image:\n{e}")

    def on_save_image(self):
        if not self._require_image():
            return
        selected_filetype = tk.StringVar(value="PNG")
        path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                
            ],
            typevariable=selected_filetype,
        )
        if not path:
            return
        if not os.path.splitext(path)[1] and selected_filetype.get() == "DICOM":
            path = f"{path}.dcm"
        self._set_status("Saving …", "busy")
        try:
            success, msg = image_io.save_image(
                self.current_image_array, path, self.metadata_dict
            )
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

        if scale > MAX_ZOOM_SCALE:
            self._set_status(
                f"Scale {scale}× exceeds the maximum allowed ({MAX_ZOOM_SCALE}×).", "error",
            )
            return

        # Pixel-count safety check — warn before freezing the app
        h, w = self.current_image_array.shape[:2]
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        if out_h * out_w > MAX_PROCESSING_PIXELS:
            msg = (
                f"The zoom result ({out_w} × {out_h} = {out_w * out_h:,} pixels) would\n"
                f"exceed the safety limit of {MAX_PROCESSING_PIXELS:,} pixels.\n\n"
                f"Please use a smaller scale factor."
            )
            self._set_status("Zoom output too large — use a smaller scale.", "error")
            messagebox.showerror("Image Too Large", msg)
            return

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

    def on_threshold(self):
        """Binarize the current image using a global intensity threshold.

        Converts grayscale (or RGB → gray) into a binary uint8 mask:
            pixel > threshold  →  255
            pixel ≤ threshold  →  0
        This is the required first step before morphological operations.
        """
        if not self._require_image():
            return

        threshold_val = int(self.threshold_slider.get())
        self._set_status(f"Applying threshold at {threshold_val} …", "busy")

        ok = self.apply_action(morphology_engine.apply_threshold, threshold_val)
        if ok:
            self._set_status(f"Threshold applied at {threshold_val}. Image is now binary.", "success")

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

        self._set_status(f"Applying {label} ({shape} SE, {size}×{size}) …", "busy")

        if operation == "erode":
            ok = self.apply_action(morphology_engine.erode, se)

        elif operation == "dilate":
            ok = self.apply_action(morphology_engine.dilate, se)

        elif operation == "open":
            # Opening = Erosion  followed by  Dilation
            ok = self.apply_action(morphology_engine.opening, se)

        elif operation == "close":
            # Closing = Dilation  followed by  Erosion
            ok = self.apply_action(morphology_engine.closing, se)

        elif operation == "boundary":
            # Boundary = current - erode(current), implemented in morphology_engine.py.
            ok = self.apply_action(morphology_engine.extract_boundary, se)

        else:
            self._set_status("Unknown morphology operation.", "error")
            return

        if ok:
            self._set_status(f"{label} applied ({shape} SE, size={size}).", "success")

    # Author: Zeyad Khaled
    def on_undo(self):
        """Undo the last pipeline step.

        History always retains the initial loaded state (index 0), so undo
        is only available when len(history) > 1.
        """
        if len(self.image_history) <= 1:
            self._set_status("No previous step to undo — already at original image.", "warning")
            return

        self.redo_history.append(self.image_history.pop().copy())  # move last result to redo stack
        self.current_image_array = self.image_history[-1].copy()  # restore previous

        h, w = self.current_image_array.shape[:2]
        self.metadata_dict["Width"]  = str(w)
        self.metadata_dict["Height"] = str(h)
        self.update_metadata_panel(self.metadata_dict)
        self.refresh_canvas(self.current_image_array)

        undoable = len(self.image_history) - 1
        self._set_status(
            f"Undo complete. {undoable} more step(s) can still be undone.", "info",
        )

    def on_redo(self):
        """Redo the most recently undone pipeline step."""
        if not self.redo_history:
            self._set_status("No step to redo.", "warning")
            return

        redo_image = self.redo_history.pop().copy()
        self.image_history.append(redo_image.copy())
        self.current_image_array = self.image_history[-1].copy()

        h, w = self.current_image_array.shape[:2]
        self.metadata_dict["Width"]  = str(w)
        self.metadata_dict["Height"] = str(h)
        self.update_metadata_panel(self.metadata_dict)
        self.refresh_canvas(self.current_image_array)
        self._set_status("Redo complete.", "info")

    # Author: Zeyad Khaled
    def on_reset(self):
        """Restore original image and clear the entire pipeline history."""
        if self.original_image_array is None:
            self._set_status("No original image to restore.", "warning")
            return

        self.current_image_array = self.original_image_array.copy()
        self.image_history       = [self.original_image_array.copy()]  # ← spec: fresh history
        self.redo_history        = []
        self.clear_roi(show_message=False)
        # Youssra-Phase2: reset restores a different image state, so template overlays are cleared.
        self.clear_template_match(show_message=False, clear_template=True)

        h, w = self.current_image_array.shape[:2]
        self.metadata_dict["Width"]  = str(w)
        self.metadata_dict["Height"] = str(h)
        self.refresh_canvas(self.current_image_array)
        self.update_metadata_panel(self.metadata_dict)
        self._set_status("Reset to original image. Pipeline history cleared.", "info")

    def toggle_split_view(self):
        if not hasattr(self, "is_split_view"):
            self.is_split_view = False
        
        self.is_split_view = not self.is_split_view
        
        if self.is_split_view:
            if not self._require_image():
                self.is_split_view = False
                return
            
            # Hide the main canvas and scrollbars
            self.canvas.grid_remove()
            self.v_scroll.grid_remove()
            self.h_scroll.grid_remove()
            
            if not hasattr(self, "split_frame") or self.split_frame is None:
                self.split_frame = ctk.CTkFrame(self.pipeline_tab, fg_color=CLR_BG_MAIN)
                self.split_frame.grid_columnconfigure(0, weight=1)
                self.split_frame.grid_columnconfigure(1, weight=1)
                self.split_frame.grid_rowconfigure(0, weight=1)
                
                self.lbl_orig = ctk.CTkLabel(self.split_frame, text="Original", font=FONT_SECTION_HDG, compound="top")
                self.lbl_orig.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
                
                self.lbl_curr = ctk.CTkLabel(self.split_frame, text="Current", font=FONT_SECTION_HDG, compound="top")
                self.lbl_curr.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
                
            self.split_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")
            self._update_split_view()
            self._set_status("Split view ON.", "info")
        else:
            if hasattr(self, "split_frame") and self.split_frame is not None:
                self.split_frame.grid_remove()
            
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.v_scroll.grid(row=0, column=1, sticky="ns")
            self.h_scroll.grid(row=1, column=0, sticky="ew")
            self.refresh_canvas(self.current_image_array)
            self._set_status("Split view OFF.", "info")

    def _update_split_view(self):
        if not getattr(self, "is_split_view", False) or not hasattr(self, "split_frame"):
            return
            
        if self.original_image_array is None or self.current_image_array is None:
            return

        def array_to_ctk(arr):
            if arr.dtype != np.uint8:
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            pil_img = Image.fromarray(arr)
            # Maximum size per half for split view while maintaining aspect ratio
            pil_img.thumbnail((500, 500), Image.LANCZOS)
            return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

        self._split_img_orig = array_to_ctk(self.original_image_array)
        self._split_img_curr = array_to_ctk(self.current_image_array)
        
        self.lbl_orig.configure(image=self._split_img_orig)
        self.lbl_curr.configure(image=self._split_img_curr)
