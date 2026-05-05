"""
ai_segmentation_engine.py
Ownership: Ahmed Hassan Bahr
Purpose: Optional inference-only AI segmentation helpers.

This module is deliberately isolated from the required image-processing
pipeline. It never writes to image_history/current_image and it treats PyTorch
and video readers as optional runtime dependencies.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from config import DEFAULT_AI_MODEL_PATH, AI_TARGET_SIZE


class AISegmentationError(RuntimeError):
    """Friendly exception for UI-safe AI segmentation failures."""


def _as_uint8_rgb(image_array):
    if image_array is None or not isinstance(image_array, np.ndarray):
        raise AISegmentationError("No valid image array was provided.")

    arr = image_array
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] < 3:
        raise AISegmentationError("Expected a grayscale or RGB image for segmentation.")

    if arr.dtype != np.uint8:
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)
    return arr[:, :, :3]


# Owner: Bahr - AI Segmentation Bonus
def load_segmentation_model(model_path=DEFAULT_AI_MODEL_PATH, framework="pytorch"):
    """Load an inference model without making PyTorch a hard app dependency."""
    if framework.lower() != "pytorch":
        raise AISegmentationError("Only PyTorch model loading is currently supported.")

    if not model_path:
        raise AISegmentationError("Please select a PyTorch model file first.")

    model_file = Path(model_path)
    if not model_file.exists():
        raise AISegmentationError(f"Model file was not found:\n{model_file}")

    try:
        import torch
    except ImportError as exc:
        raise AISegmentationError(
            "PyTorch is not installed. Install torch to enable AI segmentation."
        ) from exc

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.jit.load(str(model_file), map_location=device)
        except Exception:
            try:
                model = torch.load(str(model_file), map_location=device, weights_only=False)
            except TypeError:
                model = torch.load(str(model_file), map_location=device)

        if isinstance(model, dict):
            raise AISegmentationError(
                "This file looks like a state_dict/checkpoint. The app needs a full "
                "scripted/saved PyTorch model for inference, or the exact model "
                "architecture must be added here."
            )

        if not hasattr(model, "eval"):
            raise AISegmentationError("The selected file is not a usable PyTorch model.")

        model.to(device)
        model.eval()
        return {"model": model, "device": device, "framework": "pytorch", "path": str(model_file)}
    except AISegmentationError:
        raise
    except Exception as exc:
        raise AISegmentationError(f"Could not load the segmentation model:\n{exc}") from exc


# Owner: Bahr - AI Segmentation Bonus
def preprocess_for_segmentation(image_array, target_size=AI_TARGET_SIZE):
    """Convert an image to a normalized CHW float tensor-ready NumPy batch."""
    try:
        rgb = _as_uint8_rgb(image_array)
        pil_img = Image.fromarray(rgb)
        resized = pil_img.resize((int(target_size[0]), int(target_size[1])), Image.BILINEAR)
        arr = np.asarray(resized).astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, axis=0)
    except AISegmentationError:
        raise
    except Exception as exc:
        raise AISegmentationError(f"Could not preprocess the image:\n{exc}") from exc


# Owner: Bahr - AI Segmentation Bonus
def run_segmentation_inference(model, preprocessed_image, framework="pytorch"):
    """Run a forward pass and return the raw mask output as a NumPy array."""
    if framework.lower() != "pytorch":
        raise AISegmentationError("Only PyTorch inference is currently supported.")
    if model is None:
        raise AISegmentationError("No AI model is loaded.")
    if preprocessed_image is None:
        raise AISegmentationError("No preprocessed image was provided.")

    try:
        import torch
    except ImportError as exc:
        raise AISegmentationError(
            "PyTorch is not installed. Install torch to enable AI segmentation."
        ) from exc

    try:
        bundle = model if isinstance(model, dict) else {"model": model, "device": torch.device("cpu")}
        net = bundle["model"]
        device = bundle.get("device", torch.device("cpu"))
        x = torch.from_numpy(preprocessed_image).float().to(device)
        with torch.no_grad():
            y = net(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            if isinstance(y, dict):
                y = y["out"] if "out" in y else next(iter(y.values()))
            y = torch.sigmoid(y) if y.min() < 0 or y.max() > 1 else y
        return y.detach().cpu().numpy()
    except Exception as exc:
        raise AISegmentationError(
            "AI inference failed. The model input/output shape may not match this app.\n"
            f"Details: {exc}"
        ) from exc


# Owner: Bahr - AI Segmentation Bonus
def postprocess_segmentation_mask(raw_mask, original_shape, threshold=0.5):
    """Resize model output back to the original image and binarize it."""
    if raw_mask is None:
        raise AISegmentationError("The model did not return a segmentation mask.")

    try:
        mask = np.asarray(raw_mask)
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] <= 4 else mask[:, :, 0]
        if mask.ndim != 2:
            raise AISegmentationError(f"Unsupported mask shape from model: {mask.shape}")

        mask = mask.astype(np.float32)
        if mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min())

        out_h, out_w = int(original_shape[0]), int(original_shape[1])
        pil_mask = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8))
        pil_mask = pil_mask.resize((out_w, out_h), Image.BILINEAR)
        resized = np.asarray(pil_mask).astype(np.float32) / 255.0
        return (resized >= float(threshold)).astype(np.uint8) * 255
    except AISegmentationError:
        raise
    except Exception as exc:
        raise AISegmentationError(f"Could not postprocess the segmentation mask:\n{exc}") from exc


# Owner: Bahr - AI Segmentation Bonus
def create_segmentation_overlay(image_array, mask_array, alpha=0.4):
    """Create a cyan-green clinical overlay for a binary segmentation mask."""
    try:
        rgb = _as_uint8_rgb(image_array).astype(np.float32)
        mask = np.asarray(mask_array)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        active = mask > 0
        overlay_color = np.array([0, 230, 170], dtype=np.float32)
        out = rgb.copy()
        out[active] = (1.0 - alpha) * out[active] + alpha * overlay_color
        return out.clip(0, 255).astype(np.uint8)
    except AISegmentationError:
        raise
    except Exception as exc:
        raise AISegmentationError(f"Could not create segmentation overlay:\n{exc}") from exc


# Owner: Bahr - Data I/O
def save_segmentation_output(mask_array, overlay_array, output_dir):
    """Save the latest image mask and overlay as PNG files."""
    if mask_array is None or overlay_array is None:
        raise AISegmentationError("No segmentation result is available to save.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "polyp_segmentation_mask.png"
    overlay_path = out_dir / "polyp_segmentation_overlay.png"

    Image.fromarray(np.asarray(mask_array).astype(np.uint8)).save(mask_path)
    Image.fromarray(_as_uint8_rgb(overlay_array)).save(overlay_path)
    return {"mask": str(mask_path), "overlay": str(overlay_path)}


# Owner: Bahr - Data I/O
def load_video_file(video_path):
    """Validate a video path for optional AI frame extraction."""
    path = Path(video_path)
    if not path.exists():
        raise AISegmentationError(f"Video file was not found:\n{path}")
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise AISegmentationError("Supported video formats are mp4, avi, mov, and mkv.")
    return str(path)


# Owner: Bahr - Data I/O
def extract_video_frames(video_path, frame_step=10, max_frames=100):
    """Extract limited frames with imageio if the optional dependency exists."""
    load_video_file(video_path)
    frame_step = max(1, int(frame_step))
    max_frames = max(1, int(max_frames))

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise AISegmentationError(
            "Video segmentation requires the optional imageio dependency."
        ) from exc

    frames = []
    try:
        for frame_index, frame in enumerate(iio.imiter(video_path)):
            if frame_index % frame_step != 0:
                continue
            frames.append({"frame_index": frame_index, "image": _as_uint8_rgb(np.asarray(frame))})
            if len(frames) >= max_frames:
                break
    except Exception as exc:
        raise AISegmentationError(f"Could not read frames from the selected video:\n{exc}") from exc

    if not frames:
        raise AISegmentationError("No frames could be extracted from the selected video.")
    return frames


# Owner: Bahr - AI Segmentation Bonus
def run_video_segmentation(
    model,
    video_path,
    frame_step=10,
    max_frames=100,
    progress_callback=None,
):
    """Segment selected video frames with strict frame limits."""
    frames = extract_video_frames(video_path, frame_step=frame_step, max_frames=max_frames)
    results = []
    for idx, item in enumerate(frames, start=1):
        image = item["image"]
        preprocessed = preprocess_for_segmentation(image)
        raw_mask = run_segmentation_inference(model, preprocessed)
        mask = postprocess_segmentation_mask(raw_mask, image.shape)
        overlay = create_segmentation_overlay(image, mask)
        results.append(
            {
                "frame_index": item["frame_index"],
                "image": image,
                "mask": mask,
                "overlay": overlay,
            }
        )
        if progress_callback is not None:
            progress_callback(idx, len(frames), item["frame_index"])
    return results


# Owner: Bahr - Data I/O
def save_video_segmentation_outputs(results, output_dir):
    """Save segmented video frames and overlays as PNG files."""
    if not results:
        raise AISegmentationError("No video segmentation results are available to save.")

    out_dir = Path(output_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for item in results:
        frame_index = int(item["frame_index"])
        mask_path = masks_dir / f"frame_{frame_index:06d}_mask.png"
        overlay_path = overlays_dir / f"frame_{frame_index:06d}_overlay.png"
        Image.fromarray(np.asarray(item["mask"]).astype(np.uint8)).save(mask_path)
        Image.fromarray(_as_uint8_rgb(item["overlay"])).save(overlay_path)
        saved.append({"mask": str(mask_path), "overlay": str(overlay_path)})
    return saved
