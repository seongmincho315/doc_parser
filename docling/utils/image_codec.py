"""Small numpy<->PNG bytes helpers shared by models that call out to an
HTTP-served vision model (PaddleOCR, remote TableFormer, ...).

Extracted from PaddleOcrModel so the same encode/decode logic isn't
duplicated in every remote-model client.
"""

import io

import numpy as np
from PIL import Image


def numpy_to_image(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        a_min, a_max = float(arr.min()), float(arr.max())
        arr = (
            ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
            if a_max > a_min
            else np.zeros_like(arr, dtype=np.uint8)
        )
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """이미지를 PNG로 직렬화."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    return pil_to_png_bytes(numpy_to_image(arr))
