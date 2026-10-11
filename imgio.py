"""Shared, platform-independent image loading (phone photos, transparent PNGs, odd modes)."""
from PIL import Image, ImageOps


def load_rgb(path):
    """Open any Pillow-readable image as RGB: honours EXIF rotation (phone photos) and
    flattens transparency onto white instead of black."""
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")
