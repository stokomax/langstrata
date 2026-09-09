"""Colorize the fin logo PNG to cyan for the langstrata README."""
from PIL import Image

img = Image.open("/home/martin/projects/langstrata/docs/screenshots/fin_logo_v2.png")
img = img.convert("RGB")
w, h = img.size
pixels = img.load()

cyan = (0, 200, 220)  # bright but readable cyan
threshold = 180  # pixels with avg brightness below this get colored

for y in range(h):
    for x in range(w):
        r, g, b = pixels[x, y]
        avg = (r + g + b) / 3
        if avg < threshold:
            # Dark pixel → map to cyan, preserving darkness ratio
            t = avg / 255.0
            pixels[x, y] = (int(cyan[0] * t), int(cyan[1] * t), int(cyan[2] * t))
        else:
            # Light pixel → white
            pixels[x, y] = (255, 255, 255)

img.save("/home/martin/projects/langstrata/docs/screenshots/fin_logo_v2.png")
print("Done — saved cyan fin logo.")