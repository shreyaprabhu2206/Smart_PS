
import os
from PIL import Image, ImageEnhance, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "preprocessed")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set rotation for each image:
# 0   = No rotation
# 90  = Rotate 90 degrees clockwise
# 180 = Rotate 180 degrees
# 270 = Rotate 270 degrees clockwise

ROTATIONS = {
    "prescription_01.png": 90,
    "prescription_02.png": 90,
    "prescription_03.png": 90,
    "prescription_04.png": 90,
    "prescription_05.png": 0,
    "prescription_06.png": 0,
    "prescription_07.png": 0,
    "prescription_08.png": 90,
    "prescription_09.png": 90,
}

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        image = Image.open(input_path).convert("RGB")

        # Apply individual rotation
        angle = ROTATIONS.get(filename, 0)
        image = image.rotate(-angle, expand=True)

        # Preprocessing
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image, cutoff=1)
        image = ImageEnhance.Contrast(image).enhance(1.3)

        width, height = image.size

        image = image.resize(
            (width * 2, height * 2),
            Image.Resampling.LANCZOS
        )

        image.convert("RGB").save(output_path)

        print(f"Processed: {filename} | Rotation: {angle} degrees")

    except Exception as error:
        print("Error:", filename, error)

print("\nPreprocessing completed!")
