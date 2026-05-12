"""
ai_segmentation_engine.py
Purpose: Isolated Keras polyp segmentation inference helpers.

This module never writes to current_image_array, original_image_array, or
image_history. The UI decides whether to temporarily display results or
explicitly apply an overlay to the normal processing pipeline.
"""

# AI-Segmentation START
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_AI_CONFIG = {
    "model_file": "polyp_model.keras",
    "img_size": 256,
    "threshold": 0.5,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

_MODEL_CACHE = {}


class AISegmentationError(RuntimeError):
    """Friendly exception for UI-safe AI segmentation failures."""


def _resolve_project_path(path_like):
    """Resolve relative AI paths from cwd first, then this module directory."""
    path = Path(path_like)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    module_path = Path(__file__).resolve().parent / path
    if module_path.exists():
        return module_path

    workspace_model_path = Path(__file__).resolve().parent / "clinical_workbench" / path
    if workspace_model_path.exists():
        return workspace_model_path

    sibling_model_path = Path(__file__).resolve().parent.parent / "clinical_workbench" / path
    if sibling_model_path.exists():
        return sibling_model_path

    return module_path


def load_ai_config(config_path="ai_models/config.json"):
    """Load AI config JSON, falling back to defaults and config.json.txt if needed."""
    config_file = _resolve_project_path(config_path)
    if not config_file.exists() and config_file.suffix == ".json":
        txt_fallback = config_file.with_suffix(config_file.suffix + ".txt")
        if txt_fallback.exists():
            config_file = txt_fallback

    config = dict(DEFAULT_AI_CONFIG)
    if config_file.exists():
        try:
            with config_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception as exc:
            raise AISegmentationError(f"Could not read AI config:\n{config_file}\n{exc}") from exc

    config_dir = config_file.parent if config_file.exists() else _resolve_project_path("ai_models")
    model_path = Path(str(config.get("model_file", DEFAULT_AI_CONFIG["model_file"])))
    if not model_path.is_absolute():
        model_path = config_dir / model_path

    config["model_path"] = str(model_path)
    config["img_size"] = int(config.get("img_size", DEFAULT_AI_CONFIG["img_size"]))
    config["threshold"] = float(config.get("threshold", DEFAULT_AI_CONFIG["threshold"]))
    config["mean"] = [float(v) for v in config.get("mean", DEFAULT_AI_CONFIG["mean"])]
    config["std"] = [float(v) for v in config.get("std", DEFAULT_AI_CONFIG["std"])]
    return config


def load_model(model_path=None, config_path="ai_models/config.json"):
    """Load and cache the Keras polyp model using the original custom loss shim."""
    config = load_ai_config(config_path)
    resolved_model_path = Path(model_path) if model_path else Path(config["model_path"])
    if not resolved_model_path.is_absolute():
        resolved_model_path = _resolve_project_path(resolved_model_path)

    if not resolved_model_path.exists():
        raise AISegmentationError(f"AI model file was not found:\n{resolved_model_path}")

    cache_key = str(resolved_model_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise AISegmentationError(
            "TensorFlow is not installed. Install tensorflow to enable Keras AI segmentation."
        ) from exc

    try:
        try:
            import segmentation_models as sm

            sm.set_framework("tf.keras")
        except ImportError:
            pass

        def _dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        model = tf.keras.models.load_model(
            str(resolved_model_path),
            custom_objects={"bce_dice_loss": _dummy_loss},
            compile=False,
        )
        _MODEL_CACHE[cache_key] = model
        return model
    except Exception as exc:
        raise AISegmentationError(
            "Could not load the Keras polyp model. If it was saved with "
            "segmentation_models objects, install segmentation-models too.\n"
            f"Details: {exc}"
        ) from exc


def _to_rgb_uint8(image_array):
    """Convert grayscale/RGB/RGBA-like input to RGB uint8."""
    if image_array is None:
        raise AISegmentationError("No valid image array was provided.")

    arr = np.asarray(image_array)
    if arr.size == 0:
        raise AISegmentationError("No valid image array was provided.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr[:, :, :1], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]

    raise AISegmentationError("Expected grayscale, RGB, or RGBA image data.")


def preprocess_image(image_array, config):
    """Resize to 256x256 RGB and apply the original ImageNet-style normalization."""
    rgb = _to_rgb_uint8(image_array)
    height, width = rgb.shape[:2]
    original_size = (width, height)
    img_size = int(config.get("img_size", DEFAULT_AI_CONFIG["img_size"]))

    pil_img = Image.fromarray(rgb)
    resized = pil_img.resize((img_size, img_size), Image.BILINEAR)
    resized_rgb = np.asarray(resized).astype(np.uint8)

    mean = np.asarray(config.get("mean", DEFAULT_AI_CONFIG["mean"]), dtype=np.float32)
    std = np.asarray(config.get("std", DEFAULT_AI_CONFIG["std"]), dtype=np.float32)
    img_float = resized_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch, original_size, resized_rgb


def postprocess_mask(prediction, original_size, threshold):
    """Threshold model probability output and resize the mask to original image size."""
    pred = np.asarray(prediction)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise AISegmentationError(f"Unsupported prediction shape from model: {np.asarray(prediction).shape}")

    prob_map = pred.astype(np.float32)
    binary = prob_map > float(threshold)
    if np.any(binary):
        confidence = float(np.mean(prob_map[binary]))
    else:
        confidence = float(np.max(prob_map)) if prob_map.size else 0.0

    mask_small = (binary.astype(np.uint8) * 255)
    pil_mask = Image.fromarray(mask_small)
    pil_mask = pil_mask.resize((int(original_size[0]), int(original_size[1])), Image.NEAREST)
    return np.asarray(pil_mask).astype(np.uint8), confidence


def create_overlay(image_array, mask, color=(0, 255, 0), alpha=0.4):
    """Create a green segmentation overlay using pure NumPy blending."""
    rgb = _to_rgb_uint8(image_array).astype(np.float32)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    if mask_arr.shape[:2] != rgb.shape[:2]:
        pil_mask = Image.fromarray(np.clip(mask_arr, 0, 255).astype(np.uint8))
        pil_mask = pil_mask.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        mask_arr = np.asarray(pil_mask)

    out = rgb.copy()
    active = mask_arr > 0
    overlay_color = np.asarray(color, dtype=np.float32)
    out[active] = out[active] * (1.0 - float(alpha)) + overlay_color * float(alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def run_inference(image_array, model_path=None, config_path="ai_models/config.json", threshold=None):
    """Run full Keras polyp segmentation and return mask, overlay, and confidence."""
    config = load_ai_config(config_path)
    model = load_model(model_path=model_path, config_path=config_path)
    img_batch, original_size, _resized_rgb = preprocess_image(image_array, config)
    prediction = model.predict(img_batch, verbose=0)
    active_threshold = float(config["threshold"]) if threshold is None else float(threshold)
    mask_uint8, confidence = postprocess_mask(
        prediction,
        original_size,
        active_threshold,
    )
    overlay_rgb = create_overlay(image_array, mask_uint8, color=(0, 255, 0), alpha=0.4)
    return {
        "mask": mask_uint8,
        "overlay": overlay_rgb,
        "confidence": confidence,
        "threshold": active_threshold,
        "img_size": int(config["img_size"]),
    }


def save_segmentation_output(mask_array, overlay_array, output_dir):
    """Save mask and overlay PNG files into a selected directory."""
    if mask_array is None or overlay_array is None:
        raise AISegmentationError("No segmentation result is available to save.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "polyp_segmentation_mask.png"
    overlay_path = out_dir / "polyp_segmentation_overlay.png"
    Image.fromarray(np.asarray(mask_array).astype(np.uint8)).save(mask_path)
    Image.fromarray(_to_rgb_uint8(overlay_array)).save(overlay_path)
    return {"mask": str(mask_path), "overlay": str(overlay_path)}


# Compatibility wrappers for older UI names. They now route to the isolated Keras path.
def load_segmentation_model(model_path=None, framework="keras"):
    return load_model(model_path=model_path)


def preprocess_for_segmentation(image_array, target_size=None):
    config = dict(DEFAULT_AI_CONFIG)
    if target_size is not None:
        config["img_size"] = int(target_size[0] if isinstance(target_size, (tuple, list)) else target_size)
    batch, _original_size, _resized_rgb = preprocess_image(image_array, config)
    return batch


def run_segmentation_inference(model, preprocessed_image, framework="keras"):
    if model is None:
        raise AISegmentationError("No AI model is loaded.")
    return model.predict(preprocessed_image, verbose=0)


def postprocess_segmentation_mask(raw_mask, original_shape, threshold=0.5):
    original_size = (int(original_shape[1]), int(original_shape[0]))
    mask, _confidence = postprocess_mask(raw_mask, original_size, threshold)
    return mask


def create_segmentation_overlay(image_array, mask_array, alpha=0.4):
    return create_overlay(image_array, mask_array, color=(0, 255, 0), alpha=alpha)


# Bahr-AI-Video START
def get_video_metadata(video_path):
    """
    Read basic metadata from a video file using OpenCV.

    Returns a dict with: filename, frame_count, fps, width, height.
    Returns partial info (plus an 'error' key) on failure rather than raising.
    """
    try:
        import cv2
    except ImportError:
        return {"filename": os.path.basename(str(video_path)), "error": "OpenCV not installed"}

    meta = {"filename": os.path.basename(str(video_path))}
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            meta["fps"]         = float(cap.get(cv2.CAP_PROP_FPS))
            meta["width"]       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta["height"]      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            meta["error"] = "Could not open video"
    except Exception as exc:
        meta["error"] = str(exc)
    return meta


def segment_video_frames(
    video_path,
    model_path=None,
    config_path="ai_models/config.json",
    frame_step=5,
    max_frames=200,
    threshold=0.5,
    progress_callback=None,
):
    """
    Process a video file frame-by-frame using the Keras polyp segmentation model.

    The model is loaded once and cached (via _MODEL_CACHE) before the loop starts —
    it is never reloaded per frame.  Every frame_step-th frame is selected for
    inference; frames in between are skipped to keep memory usage bounded.

    Args:
        video_path:         Path to the input video (.mp4 / .avi / .mov / .mkv).
        model_path:         Optional override for the Keras model path.
        config_path:        Path to the AI JSON config.
        frame_step:         Process every N-th frame (1 = every frame, 5 = every 5th).
        max_frames:         Hard cap on how many frames to process (avoids OOM).
        threshold:          Segmentation probability threshold passed to run_inference.
        progress_callback:  Optional callable(current_count, estimated_total).

    Returns:
        List of dicts, one per processed frame:
            {
                "frame_index": int,          # absolute frame number in the source video
                "original":    np.uint8 RGB, # raw video frame converted BGR→RGB
                "mask":        np.uint8,     # binary segmentation mask
                "overlay":     np.uint8 RGB, # green overlay blended onto the frame
                "confidence":  float,        # mean predicted probability inside the mask
            }

    Raises:
        ValueError:           if the video cannot be opened or parameters are invalid.
        AISegmentationError:  if the Keras model fails to load.
    """
    try:
        import cv2
    except ImportError as exc:
        raise AISegmentationError(
            "OpenCV (cv2) is required for video segmentation.\n"
            "Install it with:  pip install opencv-python"
        ) from exc

    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")
    if max_frames <= 0:
        raise ValueError("max_frames must be a positive integer.")

    # Pre-load the model once so run_inference can pull it from the cache every frame
    load_model(model_path=model_path, config_path=config_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_total = min(
        max_frames,
        max(1, (total_video_frames + frame_step - 1) // frame_step),
    )

    results    = []
    frame_idx  = 0
    processed  = 0

    try:
        while processed < max_frames:
            ret, bgr_frame = cap.read()
            if not ret:
                break  # end of video or read error

            if frame_idx % frame_step == 0:
                # OpenCV produces BGR — convert to RGB before passing to the model
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                result = run_inference(
                    rgb_frame,
                    model_path=model_path,
                    config_path=config_path,
                    threshold=threshold,
                )

                results.append({
                    "frame_index": frame_idx,
                    "original":    rgb_frame,
                    "mask":        result["mask"],
                    "overlay":     result["overlay"],
                    "confidence":  result["confidence"],
                })
                processed += 1

                if progress_callback is not None:
                    try:
                        progress_callback(processed, estimated_total)
                    except Exception:
                        pass  # never let a bad callback crash the loop

            frame_idx += 1
    finally:
        cap.release()

    return results


def save_video_result(results, output_path, fps=None, original_fps=None):
    """
    Write processed overlay frames to an MP4 or AVI video file.

    Args:
        results:      List of frame dicts returned by segment_video_frames.
        output_path:  Destination file path (.mp4 or .avi).
        fps:          Output FPS.  Falls back to original_fps, then to 10.
        original_fps: FPS from the source video (used when fps is None).

    Raises:
        AISegmentationError: if results are empty or the VideoWriter cannot be created.
    """
    try:
        import cv2
    except ImportError as exc:
        raise AISegmentationError(
            "OpenCV (cv2) is required to save video output.\n"
            "Install it with:  pip install opencv-python"
        ) from exc

    if not results:
        raise AISegmentationError("No processed frames available to save.")

    out_fps = float(fps) if fps is not None else (float(original_fps) if original_fps else 10.0)
    if out_fps <= 0:
        out_fps = 10.0

    first_overlay = np.asarray(results[0]["overlay"])
    h, w = first_overlay.shape[:2]

    ext = str(output_path).lower().rsplit(".", 1)[-1]
    fourcc = cv2.VideoWriter_fourcc(*"XVID") if ext == "avi" else cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise AISegmentationError(f"Could not create video writer for: {output_path}")

    try:
        for frame_dict in results:
            overlay_rgb = np.asarray(frame_dict["overlay"]).astype(np.uint8)
            if overlay_rgb.ndim == 2:
                overlay_rgb = np.stack([overlay_rgb, overlay_rgb, overlay_rgb], axis=-1)
            # cv2.VideoWriter expects BGR
            writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


# Legacy compatibility stubs (kept so nothing that imported them breaks)
def load_video_file(video_path):
    return get_video_metadata(video_path)


def run_video_segmentation(video_path, **kwargs):
    return segment_video_frames(video_path, **kwargs)


def save_video_segmentation_outputs(results, output_path, **kwargs):
    return save_video_result(results, output_path, **kwargs)
# Bahr-AI-Video END
# AI-Segmentation END
