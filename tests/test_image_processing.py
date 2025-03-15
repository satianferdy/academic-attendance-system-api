import pytest
from app.utils.image_processing import *
from PIL import Image
import numpy as np

def test_image_conversion():
    img = Image.new('RGB', (100, 100))
    img_bytes = image_to_bytes(img)
    restored = bytes_to_image(img_bytes)
    assert restored.size == img.size

def test_image_resizing():
    img = Image.new('RGB', (200, 200))
    resized = resize_image(img, (100, 100))
    assert resized.size == (100, 100)

def test_normalization():
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    normalized = normalize_image(arr)
    assert normalized.min() >= 0
    assert normalized.max() <= 1.0