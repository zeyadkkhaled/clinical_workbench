# config.py
APP_TITLE       = "Clinical Image Analysis Workbench"
WINDOW_GEOMETRY = "1400x900"

THEME_COLOR     = "dark-blue"
APPEARANCE_MODE = "dark"

UI_PADDING    = 10
CORNER_RADIUS = 8
SIDEBAR_WIDTH = 290

# Safety limits
MAX_PROCESSING_PIXELS = 40_000_000   # 40 MP — blocks catastrophically large zoom outputs
MAX_ZOOM_SCALE        = 16.0         # hard cap on zoom scale factor

# Owner: Bahr - AI Segmentation Bonus
# Optional inference-only AI defaults. These do not affect the core pipeline.
DEFAULT_AI_MODEL_PATH = ""
AI_TARGET_SIZE        = (256, 256)
AI_VIDEO_FRAME_STEP   = 10
AI_VIDEO_MAX_FRAMES   = 100

# ── Color palette ──────────────────────────────────────────────────────────────
CLR_BG_MAIN     = "#0d1117"
CLR_BG_SIDEBAR  = "#161b22"
CLR_BG_CARD     = "#1c2128"
CLR_BG_CARD_HDR = "#21262d"
CLR_BORDER      = "#30363d"

CLR_ACCENT      = "#2563eb"
CLR_ACCENT_HOVER = "#1d4ed8"
CLR_ACCENT_DIM  = "#1e3058"

CLR_CYAN        = "#0891b2"
CLR_CYAN_HOVER  = "#0e7490"
CLR_AI_ACCENT   = "#00e6aa"
CLR_AI_PANEL    = "#071821"
CLR_AI_CARD     = "#0b2230"

CLR_TEXT_PRI    = "#e6edf3"
CLR_TEXT_SEC    = "#8b949e"
CLR_TEXT_HDG    = "#58a6ff"

CLR_SUCCESS     = "#238636"
CLR_SUCCESS_HVR = "#2ea043"
CLR_WARNING     = "#9e6a03"
CLR_WARNING_HVR = "#b07d0c"
CLR_DANGER      = "#b91c1c"
CLR_DANGER_HVR  = "#dc2626"

# Status bar background tints
CLR_ST_IDLE    = "#161b22"
CLR_ST_SUCCESS = "#0f2a18"
CLR_ST_WARNING = "#2a1f00"
CLR_ST_ERROR   = "#2a0f0f"
CLR_ST_BUSY    = "#0f1f3a"

# ── Typography ─────────────────────────────────────────────────────────────────
FONT_APP_TITLE   = ("Segoe UI", 13, "bold")
FONT_SECTION_HDG = ("Segoe UI", 11, "bold")
FONT_LABEL       = ("Segoe UI", 10)
FONT_LABEL_BOLD  = ("Segoe UI", 10, "bold")
FONT_BUTTON      = ("Segoe UI", 10, "bold")
FONT_MONO        = ("Consolas", 9)
FONT_STATUS      = ("Segoe UI", 10)
