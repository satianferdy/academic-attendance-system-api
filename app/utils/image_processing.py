from PIL import Image
import io
import numpy as np
import base64

def read_image_file(file):
    """
    Read an image file to a PIL Image.
    
    Args:
        file: File-like object or path to image file
        
    Returns:
        PIL.Image: Loaded image
    """
    if isinstance(file, str):
        return Image.open(file)
    else:
        return Image.open(file)

def image_to_bytes(image, format='JPEG'):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image
        format: Image format (default: JPEG)
        
    Returns:
        bytes: Image data as bytes
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=format)
    return img_bytes.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(io.BytesIO(image_bytes))

def resize_image(image, size):
    """
    Resize an image to the specified size.
    
    Args:
        image: PIL Image
        size: Tuple of (width, height)
        
    Returns:
        PIL.Image: Resized image
    """
    return image.resize(size, Image.LANCZOS)

def normalize_image(image_array):
    """
    Normalize image pixel values to the range [0, 1].
    
    Args:
        image_array: Numpy array of image
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    return image_array / 255.0

def image_to_base64(image, format='JPEG'):
    """
    Convert a PIL Image to a base64 string.
    
    Args:
        image: PIL Image
        format: Image format (default: JPEG)
        
    Returns:
        str: Base64-encoded image string
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=format)
    img_bytes = img_bytes.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def base64_to_image(base64_string):
    """
    Convert a base64 string to a PIL Image.
    
    Args:
        base64_string: Base64-encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes))