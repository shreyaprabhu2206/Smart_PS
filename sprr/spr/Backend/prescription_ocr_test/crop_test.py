
import os
from PIL import Image, ImageOps

INPUT_FOLDER = "preprocessed"
OUTPUT_FOLDER = "cropped"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(INPUT_FOLDER, filename)

    image = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(image)

    # Identify darker pixels belonging to handwriting
    mask = gray.point(lambda pixel: 255 if pixel < 170 else 0)

    # Find the bounding box around the handwriting
    bbox = mask.getbbox()

    if bbox:
        left, top, right, bottom = bbox

        # Add padding around the handwriting
        padding = 40

        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)

        cropped = image.crop((left, top, right, bottom))

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cropped.save(output_path)

        print("Cropped:", filename)

print("Cropping completed!")
